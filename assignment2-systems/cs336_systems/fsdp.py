import torch
from torch import nn
from torch import distributed as dist
from itertools import pairwise
import cs336_basics.model as mdl

compatible_modules = (mdl.Linear, mdl.Embedding, nn.Linear, nn.Embedding)

class FSDP(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.shard_dict = {}
        self.pad_len = {}
        self.full_chunk_size = {}
        self.compute_dtype = compute_dtype
        hooked_on_modules: list[nn.Module] = []
        self.non_sharded_modules = []
        self.async_handles: dict[torch.nn.Module, list[dist.Work | None]] = {}
        self.execution_order = []
        self.order_established = False
        for module in self.module.modules():
            self.async_handles[module] = []
            if isinstance(module, compatible_modules):
                self.shard(module)
                hooked_on_modules.append(module)
            else:
                for param in module.parameters(recurse=False):
                    if param.requires_grad:
                        param.register_post_accumulate_grad_hook(self.sync_nonsharded_grads(param))
                self.non_sharded_modules.append(module)
        self.total_hooked_modules = sum(1 for _ in hooked_on_modules)
        if len(hooked_on_modules) < 2:
            print("WARNING: using FSDP without any pairs of compatible modules, no hooks will happen")
            return

        for l1 in hooked_on_modules:
            self.load_layer_hook(l1)
            self.sync_weights_hook(l1)
            self.unload_layer_hook(l1)

            self.backwards_sync_weights_hook(l1)    
            self.backward_load_layer_hook(l1)
            for param in l1.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(self.reduce_scatter_unload(param))
    
    def forward(self, *inputs, **kwargs):
        # with torch.autocast(inputs[0].device, self.compute_dtype):
        return self.module(*inputs, **kwargs)  
        
    def shard(self, module: nn.Module):
        for parameter in module.parameters():
            chunks = torch.chunk(parameter.data, dist.get_world_size())
            unpadded_last_chunk_size = chunks[-1].shape[0]
            last_chunk = torch.zeros_like(chunks[0])
            self.pad_len[parameter] = (chunks[0].shape[0] - chunks[-1].shape[0]) # torch.chunk splits by dim 0
            self.full_chunk_size[parameter] = chunks[0].shape
            self.shard_dict[parameter] = chunks[dist.get_rank()].clone()

            if dist.get_rank() == dist.get_world_size() - 1: # this is the last chunk (world_size 1 only has rank 0)
                if self.pad_len[parameter] != 0:
                    last_chunk[:unpadded_last_chunk_size, :] = self.shard_dict[parameter]
                    assert last_chunk.shape == chunks[0].shape
                    self.shard_dict[parameter] = last_chunk
    
    def sync_nonsharded_grads(self, param):
        def hook(param):
            grad = param.grad.contiguous() / dist.get_world_size()
            dist.all_reduce(grad)
            param.grad = grad 
            
        return hook
    
    def load_mod_helper(self, mods: nn.Module, async_op, full_params=False):
        for module in mods.modules():
            for parameter in module.parameters():
                cast_shard = self.shard_dict[parameter]
                weights = torch.cat([torch.zeros_like(cast_shard) for _ in range(dist.get_world_size())])
                if not full_params:
                    cast_shard = cast_shard.to(self.compute_dtype)
                    weights = weights.to(self.compute_dtype)
                if async_op:
                    self.async_handles[module].append(dist.all_gather_into_tensor(weights, cast_shard, async_op=async_op))
                else:
                    dist.all_gather_into_tensor(weights, cast_shard, async_op=async_op)
                if self.pad_len[parameter] != 0:
                    parameter.data = weights[:-self.pad_len[parameter], ...]
                else:
                    parameter.data = weights        
                    
    def gather_params(self, current_module, async_op=False, full_params=False):
        if not self.order_established: # first pass through, load self
            self.execution_order.append(current_module)
            self.load_mod_helper(current_module, async_op, full_params)
            if len(self.execution_order) == self.total_hooked_modules: # execution order is established
                self.order_established = True
            return
        if current_module == self.execution_order[0]: # always load the first module
            self.load_mod_helper(current_module, async_op, full_params) 
        module_to_load_idx = self.execution_order.index(current_module) + 1
        if module_to_load_idx == self.total_hooked_modules: # it is the last module
            return 
        module_to_load = self.execution_order[module_to_load_idx]
        self.load_mod_helper(module_to_load, async_op, full_params)
    
            
    def gather_params_backward(self, current_module):
        if current_module == self.execution_order[-1]: # always load the last layer
            self.load_mod_helper(current_module, async_op=True, full_params=False)
        if current_module == self.execution_order[0]: # no-op
            return
        module_to_load_idx = self.execution_order.index(current_module) - 1
        module_to_load = self.execution_order[module_to_load_idx]
        self.load_mod_helper(module_to_load, async_op=True, full_params=False)
        
    def reduce_scatter_unload(self, _: nn.Parameter):
        def hook(param: nn.Parameter) -> None:
            if not param.requires_grad:
                raise ValueError("huh?")

            param.grad /= dist.get_world_size()
            desired_size =  self.full_chunk_size[param][0] * dist.get_world_size()
            grad_inp = torch.zeros((desired_size, *param.shape[1:]), device=param.device)
            grad_inp[:param.shape[0], ...] = param.grad
            output_buffer = torch.zeros_like(self.shard_dict[param])
            dist.reduce_scatter_tensor(output_buffer, grad_inp)
            param.data = self.shard_dict[param]
            param.grad = output_buffer
            
        return hook
    # unload the current module
    def unload_module(self, _: nn.Module):
        last_module = list(self.module.modules())[-1]
        def hook(l1: nn.Module, args, output):
            for parameter in l1.parameters():
                parameter.data = self.shard_dict[parameter]
        return hook

    def gather_full_params(self):
        for module in self.module.modules():
            if isinstance(module, compatible_modules):
                self.gather_params(module, full_params=True)
        state_dict = self.module.state_dict().copy()
        for module in self.module.modules():
            if isinstance(module, compatible_modules):
                for parameter in module.parameters():
                    parameter.data = self.shard_dict[parameter]
        return state_dict    
    
    def sync_weights(self, module: torch.nn.Module):
        for i, handle in enumerate(self.async_handles[module]):
            if handle is None:
                continue
            handle.wait()
        self.async_handles[module].clear()
    
    def finish_gradient_synchronization(self):
        for handles in self.async_handles.values():
            for handle in handles:
                if handle is None:
                    continue
                handle.wait()
        
    def load_layer_hook(self, l1: torch.nn.Module):
        l1.register_forward_pre_hook(lambda l, _: self.gather_params(l, async_op=True))
    
    def backward_load_layer_hook(self, l1: torch.nn.Module):
        l1.register_full_backward_pre_hook(lambda _1, _2: self.gather_params_backward(_1))

    def backwards_sync_weights_hook(self, l1: torch.nn.Module):
        l1.register_full_backward_pre_hook(lambda _0,_1: self.sync_weights(l1)) # I tried this both after and before the other hook

    def sync_weights_hook(self, l1: torch.nn.Module):
        l1.register_forward_pre_hook(lambda _0, _1: self.sync_weights(l1))
        
    def unload_layer_hook(self, l1: torch.nn.Module):
        l1.register_forward_hook(self.unload_module(l1))