import torch
import torch.distributed as dist
from typing import Any, Type
from torch.optim import Optimizer



class NaiveDistributor(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        if torch.cuda.is_available():
            device = f"cuda:{dist.get_rank()}"
            self.module.to(device)
        params = torch.nn.utils.parameters_to_vector(self.module.parameters())
        dist.broadcast(params, src=0)
        torch.nn.utils.vector_to_parameters(params, self.module.parameters())
        
    def forward(self, X):
        return self.module(X)
    
    def sync_weights(self):
        params = torch.nn.utils.parameters_to_vector(self.module.parameters())
        dist.broadcast(params, src=0)
        torch.nn.utils.vector_to_parameters(params, self.module.parameters())
    
    def ddp_on_after_backward(self, optimizer):
        self.sync()
        
    def sync(self):
        for p in self.module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad /= dist.get_world_size()
                
class FlatDistributor(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        if torch.cuda.is_available():
            device = f"cuda:{dist.get_rank()}"
            self.module.to(device)
        params = torch.nn.utils.parameters_to_vector(self.module.parameters())
        dist.broadcast(params, src=0)
        torch.nn.utils.vector_to_parameters(params, self.module.parameters())
        
    def forward(self, X):
        return self.module(X)
    
    def sync_weights(self):
        params = torch.nn.utils.parameters_to_vector(self.module.parameters())
        dist.broadcast(params, src=0)
        torch.nn.utils.vector_to_parameters(params, self.module.parameters())
    
    def ddp_on_after_backward(self, optimizer):
        self.sync()
        
    def sync(self):
        grads = [p.grad for p in self.module.parameters() if p.grad is not None]
        flat_grads = torch._utils._flatten_dense_tensors(grads)
        flat_grads /=  dist.get_world_size()
        dist.all_reduce(flat_grads)
        params = [p for p in self.module.parameters() if p.grad is not None]
        unflat_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)
        for g, p in zip(unflat_grads, params):
            p.grad = g
            
class OverlappingDistributor(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        if torch.cuda.is_available():
            device = f"cuda:{dist.get_rank()}"
            self.module.to(device)
        self.hook_handles = []
        self.async_handles = []
        self.debug_print_done = False
        params = torch.nn.utils.parameters_to_vector(self.module.parameters())
        dist.broadcast(params, src=0)
        torch.nn.utils.vector_to_parameters(params, self.module.parameters())
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                post_handle = param.register_post_accumulate_grad_hook(self.make_post_hook(param))
    
    def make_post_hook(self, param):
        def hook(param):
            grad = param.grad.contiguous() / dist.get_world_size()
            self.async_handles.append(dist.all_reduce(grad, async_op=True))
            param.grad = grad 
            
        return hook
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    def ddp_on_after_backward(self, optimizer):
        # if not self.debug_print_done:
            # print("Just waiting on sync")
            # self.debug_print_done = True
        self.finish_gradient_synchronization()

    def finish_gradient_synchronization(self):
        for handle in self.async_handles:
            handle.wait()
        self.async_handles.clear()
        
class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        all_params: list[torch.nn.Parameter] = list(params)
        self.all_params = all_params
        self.rank = dist.get_rank()
        self.sharded_idxs = ShardedOptimizer.get_param_idx(self.all_params)
        self.responsible_params: list[torch.nn.Parameter] = [self.all_params[i] for i in self.sharded_idxs[self.rank]]

        self.optim = None
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
      
        super().__init__(self.all_params, kwargs)
    
    @staticmethod
    def get_param_idx(params):
        world_size = dist.get_world_size() # if world_size isn't None, I'm just testing this for debugging
        worker_idxs = [[] for _ in range(world_size)]
        worker_alloc: list[int] = [0  for _ in range(world_size)]
        param_summary = []
        for i, param in enumerate(params):
            param_summary.append((i, param.numel()))
        param_summary = sorted(param_summary, key=lambda v: v[1])
        while param_summary != []:
            pair = param_summary.pop()
            idx: int = pair[0]
            numel: int = pair[1]
            freest_worker = min(worker_alloc)
            min_idx: int = worker_alloc.index(freest_worker)
            worker_idxs[min_idx].append(idx)
            worker_alloc[min_idx] += numel
        return worker_idxs # the indices worker with rank `rank` are responsible for
    
    def step(self, closure=None, **kwargs):
        self.optim.step(closure=closure, **kwargs)
        for rank, param_idxs in enumerate(self.sharded_idxs):
            for param_idx in param_idxs:
                with torch.no_grad():
                    dist.broadcast(self.all_params[param_idx], src=rank)

    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)
        responsible_group = {"params": []}
        for param in param_group["params"]:
            if any(param is t for t in self.responsible_params):
                responsible_group["params"].append(param)
        if self.optim is None:
            self.optim = self.optimizer_cls([responsible_group], **self.kwargs)
            del self.kwargs, self.optimizer_cls
        else:
            self.optim.add_param_group(responsible_group)
        # self.param_groups.append(param_group)