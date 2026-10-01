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
        
        for module in self.module.modules():
            if isinstance(module, compatible_modules):
                # COMMUNICATION HERE 
                self.shard(module)
                hooked_on_modules.append(module)
            else:
                for param in module.parameters(recurse=False):
                    if param.requires_grad:
                        param.register_post_accumulate_grad_hook(self.sync_nonsharded_grads(param))
                self.non_sharded_modules.append(module)
        if len(hooked_on_modules) < 2:
            print("WARNING: using FSDP without any pairs of compatible modules, no hooks will happen")
            return
        l0 = hooked_on_modules[0]
        l0.register_forward_pre_hook(self.load_layer(l0))
        for param in l0.parameters():
            param.register_post_accumulate_grad_hook(self.reduce_scatter_unload(param))
        l_last: nn.Module = hooked_on_modules[-1]
        l_last.register_full_backward_pre_hook(self.load_layer(l_last))
        for l1, l2 in pairwise(hooked_on_modules):
            l1.register_forward_pre_hook(self.load_layer(l2)) # load next modules weight before current forward
            l1.register_forward_hook(self.unload_layer(l1)) # unload this modules weight after forward
    
            l2.register_full_backward_pre_hook(self.load_layer(l1))
            for param in l2.parameters():
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
            self.shard_dict[parameter] = chunks[dist.get_rank()]

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
    
    # load the module of then next layer
    def load_layer(self, l2: nn.Module):
        def hook(_: nn.Module, args):
            for module in l2.modules():
                for parameter in module.parameters():
                    weights = torch.cat([torch.zeros_like(self.shard_dict[parameter]) for _ in range(dist.get_world_size())]).to(self.compute_dtype)
                    cast_shard = self.shard_dict[parameter].to(self.compute_dtype)
                    dist.all_gather_into_tensor(weights, cast_shard)
                    if self.pad_len[parameter] != 0:
                        parameter.data = weights[:-self.pad_len[parameter], ...]
                    else:
                        parameter.data = weights
        return hook

    def _gather(self, outer_module):
        for module in outer_module.modules():
            for parameter in module.parameters():
                cast_shard = self.shard_dict[parameter]
                weights = torch.cat([torch.zeros_like(cast_shard) for _ in range(dist.get_world_size())])
                dist.all_gather_into_tensor(weights, cast_shard)
                if self.pad_len[parameter] != 0:
                    parameter.data = weights[:-self.pad_len[parameter], ...]
                else:
                    parameter.data = weights

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
    def unload_layer(self, _: nn.Module):
        def hook(l1: nn.Module, args, output):
            for parameter in l1.parameters():
                parameter.data = self.shard_dict[parameter]
        return hook

    def gather_full_params(self):
        for module in self.module.modules():
            if isinstance(module, compatible_modules):
                self._gather(module)
        state_dict = self.module.state_dict().copy()
        for module in self.module.modules():
            if isinstance(module, compatible_modules):
                for parameter in module.parameters():
                    parameter.data = self.shard_dict[parameter]
        return state_dict    
            


    
    def finish_gradient_synchronization(self):
        pass
        
        