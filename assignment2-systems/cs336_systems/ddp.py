import torch
import torch.distributed as dist

class NaiveDistributor(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, device: str | None =None):
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
    def __init__(self, module: torch.nn.Module, device: str | None =None):
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