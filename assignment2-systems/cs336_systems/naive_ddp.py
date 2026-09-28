import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from einops import rearrange

from tests.common import ToyModel
# from torchvision.models import mobilenet_v3_small # <- in case we want a "simpler" model
import argparse

argp = argparse.ArgumentParser()
worker_choices = [2,4,6]
argp.add_argument("--num_workers", type=int, default=4, choices=worker_choices)

argp.add_argument("--loop", action="store_true")

num_floats = { 
             "1MB": 2**20//4,
             "10MB": 2**20//4*10,
             "100MB": 2**20//4*100,
             "1GB": 2**28 # 2^30 / 4
}

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

class NaiveDistributor(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        params = torch.nn.utils.parameters_to_vector(self.module.parameters())
        dist.broadcast(params, src=0)
        torch.nn.utils.vector_to_parameters(params, self.module.parameters())
        
    def forward(self, X):
        return self.module(X)
    
    def ddp_on_after_backward(self, optimizer):
        self.sync()
        
    def sync(self):
        for p in self.module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad /= dist.get_world_size()
        

# def distributed_demo(rank, world_size, x_full: torch.Tensor, y_full: torch.Tensor, model_class: type[torch.nn.Module], loss_fn, queue: mp.Queue, device="cpu", init_dict=None):
#     setup(rank, world_size)
#     assert x_full.numel() % world_size == 0, "Batch should be divisible by the total number of workers"
#     X_chunk = torch.chunk(x_full, world_size)[rank]
#     Y_chunk = torch.chunk(y_full, world_size)[rank]
    
#     if "cuda" in device:
#         device = f"cuda:{rank}"
#     torch.random.manual_seed(0)
#     if init_dict:
#         model: torch.nn.Module = model_class(**init_dict) # number of heads, etc
#     else:
#         model: torch.nn.Module = model_class() # no parameters needed
#     model.to(device)
#     optim = torch.optim.AdamW(model.parameters())
#     params = torch.nn.utils.parameters_to_vector(model.parameters())
#     dist.broadcast(params, src=0)
#     torch.nn.utils.vector_to_parameters(params, model.parameters())
#     for _ in range(50):
#         torch.manual_seed(0)
#         y_pred = model(X_chunk)
#         loss = loss_fn(y_pred, Y_chunk)
#         loss.backward()
#         # Gradient sync
#         for p in model.parameters():
#             if p.grad is not None:
#                 dist.all_reduce(p.grad)
#                 p.grad /= world_size
#         optim.step()
#         optim.zero_grad()
#     if rank == 0:
#         queue.put(model)


# def main():
#     world_size = 2
#     # ToyModel
#     d_model = 10  
#     n_samples = 20
#     DATA = torch.randn((n_samples+1, d_model))
#     X = DATA[:-1, :]
#     Y = DATA[1:, :] # really basic targets, clone for redundancy
#     queue = mp.Queue()
#     loss_fn = torch.nn.CrossEntropyLoss()
#     res = mp.spawn(fn=distributed_demo, args=(world_size, X, Y, ToyModel, loss_fn, queue,"cpu", None), nprocs=world_size, join=True)
#     mp_model = queue.get()
#     torch.random.manual_seed(0)
#     serial_model = ToyModel()
#     optim = torch.optim.AdamW(serial_model.parameters())
#     for _ in range(50):
#         torch.manual_seed(0)
#         y_pred = serial_model(X)
#         loss = loss_fn(y_pred, Y)
#         loss.backward()
#         optim.step()
#         optim.zero_grad()
    
#     for p1, p2 in zip(mp_model.parameters(), serial_model.parameters()):
#         if not torch.allclose(p1, p2):
#             print((p1 - p2).abs().max())
#             print("Not equal")
#             return
#     print("Checks out")
    
    
# if __name__ == "__main__":
#     main()