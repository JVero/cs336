import os
import torch
from torch import distributed as dist
from torch import multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.ddp import ShardedOptimizer, OverlappingDistributor
from cs336_basics.nn_utils import cross_entropy

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
def prof_test(rank, world_size, backend, reading):
    setup(rank, world_size, backend)
    if backend == "nccl":
        device = f"cuda:{dist.get_rank()}"
        torch.set_default_device(device)
        inner_model = BasicsTransformerLM(vocab_size=10000, context_length=512, d_model=2560,
                num_layers=32, num_heads=32, d_ff=10240)
        model = OverlappingDistributor(inner_model)
        X = torch.randint(0, 10000, (1, 512,))
        Y = torch.randint(0, 10000, (1, 512,))
    else:
        device = None
        inner_model = BasicsTransformerLM(10, 10, 10, 10, 1, 10)
        model = OverlappingDistributor(inner_model)
        X = torch.randint(0, 9, (1, 10))
        Y = torch.randint(0, 9, (1, 10))
    optim = ShardedOptimizer(model.parameters(), torch.optim.AdamW)
    if torch.cuda.is_available():
        print(f"rank {dist.get_rank()} step {reading} post-init memory: {torch.cuda.max_memory_allocated(device)}")        
    y_pred = model(X)
    loss = cross_entropy(y_pred, Y)
    loss.backward()
    if torch.cuda.is_available():
        print(f"rank {dist.get_rank()} step {reading} sharded Pre optimizer step: {torch.cuda.max_memory_allocated(device)}")    
    optim.step()
    if torch.cuda.is_available():
        print(f"rank {dist.get_rank()} step {reading} sharded Post optimizer step: {torch.cuda.max_memory_allocated(device)}")    
        torch.cuda.reset_max_memory_allocated()

if __name__ == "__main__":
    world_size = 2
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    for reading in range(3):
        mp.spawn(prof_test, (world_size, backend, reading), world_size, join=True)
        if backend == "nccl":
            model = BasicsTransformerLM(vocab_size=10000, context_length=512, d_model=2560,
                    num_layers=32, num_heads=32, d_ff=10240)
            X = torch.randint(0, 10000, (1, 512,))
            Y = torch.randint(0, 10000, (1, 512,))
        else:
            model = BasicsTransformerLM(10, 10, 10, 10, 1, 10)
            X = torch.randint(0, 9, (1, 10))
            Y = torch.randint(0, 9, (1, 10))
        optim = torch.optim.AdamW(model.parameters())
        if torch.cuda.is_available():
            print(f"serial step {reading} post-init memory: {torch.cuda.max_memory_allocated()}")     
        y_pred = model(X)
        loss = cross_entropy(y_pred, Y)
        loss.backward()
        if torch.cuda.is_available():
            print(f"step {reading} serial Pre optimizer step: {torch.cuda.max_memory_allocated()}")    
        optim.step()
        if torch.cuda.is_available():
            print(f"step {reading} serial Post optimizer step: {torch.cuda.max_memory_allocated()}")    
            torch.cuda.reset_max_memory_allocated()