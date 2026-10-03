import os
import torch
from torch import distributed as dist
from torch import multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.ddp import ShardedOptimizer, OverlappingDistributor
from cs336_basics.nn_utils import cross_entropy
import contextlib
import time

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
@contextlib.contextmanager
def mytimer(times):
    start_time = time.perf_counter_ns()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter_ns()
        times.append(torch.Tensor([end_time-start_time]))
    
def warmup(model: torch.nn.Module, X: torch.Tensor, optimizer: torch.optim.Optimizer, n_steps=3):
    for _ in range(n_steps):
        y = model(X)
        y.sum().backward()
        optimizer.step()

def sharded_optim_run(rank, world_size, backend, reading):
    setup(rank, world_size, backend)
    fwd_times = []
    bac_times = []
    step_times = []
    full_times = []
    if backend == "nccl":
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)
        torch.set_default_device(device)
        inner_model = BasicsTransformerLM(vocab_size=10000, context_length=512, d_model=2560,
                num_layers=32, num_heads=32, d_ff=10240)
        model = OverlappingDistributor(inner_model) # autopicks device = cuda
        X = torch.randint(0, 10000, (1, 512,))
        Y = torch.randint(0, 10000, (1, 512,))
    else:
        device = None
        inner_model = BasicsTransformerLM(10, 10, 10, 10, 1, 10)
        model = OverlappingDistributor(inner_model) # autopicks device = cpu
        X = torch.randint(0, 9, (1, 10))
        Y = torch.randint(0, 9, (1, 10))
    optim = ShardedOptimizer(model.parameters(), torch.optim.AdamW)
    warmup(model, X, optim)
    with mytimer(full_times):
        with mytimer(fwd_times):
            y_pred = model(X)
        loss = cross_entropy(y_pred, Y)
        with mytimer(bac_times):
            loss.backward()
            model.finish_gradient_synchronization()
        with mytimer(step_times):
            optim.step()
    if dist.get_rank() == 0:
        for label, t in [("fwd", fwd_times), ("bac", bac_times), ("step", step_times), ("full", full_times)]:
            print(f"sharded {reading}: {label} times: {t}")
    dist.destroy_process_group()

def distributed_optim_run(rank, world_size, backend, reading):
    setup(rank, world_size, backend)            
    fwd_times = []
    bac_times = []
    step_times = []
    full_times = []
    if backend == "nccl":
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)
        torch.set_default_device(device)
        inner_model = BasicsTransformerLM(vocab_size=10000, context_length=512, d_model=2560,
                num_layers=32, num_heads=32, d_ff=10240)
        model = OverlappingDistributor(inner_model)
        X = torch.randint(0, 10000, (1, 512,))
        Y = torch.randint(0, 10000, (1, 512,))
    else:
        inner_model = BasicsTransformerLM(10, 10, 10, 10, 1, 10)
        model = OverlappingDistributor(inner_model)
        X = torch.randint(0, 9, (1, 10))
        Y = torch.randint(0, 9, (1, 10))
    optim = torch.optim.AdamW(model.parameters())
    warmup(model, X, optim)
    with mytimer(full_times):
        with mytimer(fwd_times):
            y_pred = model(X)
        loss = cross_entropy(y_pred, Y)
        with mytimer(bac_times):
            loss.backward()
            model.finish_gradient_synchronization()
        with mytimer(step_times):
            optim.step()
    if dist.get_rank() == 0:
        for label, t in [("fwd", fwd_times), ("bac", bac_times), ("step", step_times), ("full", full_times)]:
            print(f"serial {reading}: {label} times: {t}")
    dist.destroy_process_group()
    
if __name__ == "__main__":
    world_size = 2
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    for reading in range(3):
        mp.spawn(sharded_optim_run, (world_size, backend, reading), world_size, join=True)
    for reading in range(3):
        mp.spawn(distributed_optim_run, (world_size, backend, reading), world_size, join=True)