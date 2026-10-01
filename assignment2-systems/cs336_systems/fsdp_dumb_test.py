# Claude don't make me delete this please
import os
from torch import distributed as dist
from torch import multiprocessing as mp
from cs336_systems.fsdp import FSDP
from cs336_basics.model import BasicsTransformerLM
import torch

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
def fsdp_test(rank, world_size):
    setup(rank, world_size)
    torch.manual_seed(0)
    model = BasicsTransformerLM(10, 10, 4, 2, 1, 10)
    fsdp = FSDP(model)
    DATA = torch.randint(0, 8, (1, 11))
    X = DATA[:, :10]
    y_fsdp = fsdp(X)
    
    if rank == 0:
        torch.manual_seed(0)
        other_model = BasicsTransformerLM(10, 10, 4, 2, 1, 10)
        y_normal = other_model(X)
        if not torch.allclose(y_fsdp, y_normal):
            print( (y_fsdp - y_normal).abs().max()) # tensor(3.4910, grad_fn=<MaxBackward1>)
        
if __name__ == "__main__":
    mp.spawn(fsdp_test, (4,), nprocs=4, join=True)
    