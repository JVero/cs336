import os
import torch
import torch.distributed as dist
import contextlib

import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from tests.common import ToyModel

from cs336_systems.ddp import *

import time
import numpy as np

import argparse
param_options = {
    "xl": {
        "vocab_size": 10000,
        "context_length": 512,
        "d_model": 2560,
        "num_layers": 32,
        "num_heads": 32,
        "d_ff": 10240,
    },
    "toy": {
        "vocab_size": 100,
        "context_length": 2,
        "d_model": 2**8,
        "num_layers": 2,
        "num_heads": 32,
        "d_ff": 10,
    }
}

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
    
cls_options = {
    "naive": NaiveDistributor,
    "flat": FlatDistributor
}

backend_options = ["gloo", "nccl"]
argparser = argparse.ArgumentParser()
argparser.add_argument("--backend", choices=backend_options, default="gloo")
argparser.add_argument("--modelsize", choices=param_options.keys(), default="xl")
argparser.add_argument("--ddp", choices=list(cls_options.keys()), default="naive")
argparser.add_argument("--warmupsteps", type=int, default=3)

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def log_tensor_list(x: list[torch.Tensor], label, device):
    tensors = torch.cat(x).to(device)
    dist.all_reduce(tensors)
    if dist.get_rank() == 0:
        mean = tensors.mean() / dist.get_world_size()
        print(f"{label} mean time: {mean}")
        return mean
        
def distributed_prof(rank, world_size, X_full: torch.Tensor, Y_full: torch.Tensor,
                     model_class: type[torch.nn.Module], backend, warmup_count, ddp_type, init_dict=None):
    setup(rank, world_size, backend)
    assert X_full.numel() % world_size == 0, "Batch should be divisible by the total number of workers"
    X_chunk = torch.chunk(X_full, world_size)[rank]
    Y_chunk = torch.chunk(Y_full, world_size)[rank]
        
    if isinstance(init_dict, dict):
        model = ddp_type(model_class(**init_dict))
    else:
        model = ddp_type(model_class())

    if torch.cuda.is_available():
        device = f"cuda:{rank}"
    else:
        device = "cpu"
    model.to(device)
    X_chunk = X_chunk.to(device)
    Y_chunk = Y_chunk.to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

    optim = torch.optim.AdamW(model.parameters())
    loss = cross_entropy
    # Warmup
    with mytimer([]):
        for _ in range(warmup_count):
            y = model(X_chunk)
            loss_val = loss(y, Y_chunk)
            loss_val.backward()
            model.ddp_on_after_backward(optim)
            optim.step()
            optim.zero_grad()
    
    # Clears the previous gradient values, but keeps the graphs
    
    fwd_times = []
    bck_times = []
    full_step_times = []
    grad_sync_times = []
    opt_times = []
    n_steps = 10
    for _ in range(n_steps):
        dist.barrier()
        with mytimer(full_step_times):

            with mytimer(fwd_times):
                y_pred = model(X_chunk)
            
            with mytimer(bck_times):
                loss_val = loss(y_pred, Y_chunk)
                loss_val.backward()

            with mytimer(grad_sync_times):
                model.ddp_on_after_backward(optim)
                
            with mytimer(opt_times):
                optim.step()
                optim.zero_grad()

    mst = log_tensor_list(full_step_times, "Full step times", device)
    mft = log_tensor_list(fwd_times, "Forward step times", device)
    bck = log_tensor_list(bck_times, "Backward step times", device)
    mgt = log_tensor_list(grad_sync_times, "Mean time averaging gradients", device)
    opt = log_tensor_list(opt_times, "Mean optimizer step time", device)
    
    if rank == 0:
        assert mst is not None and mgt is not None and mft is not None
        assert bck is not None and opt is not None
        print(f"Percent of full step: \nGrad: {100 * mgt / mst}%\nFwd: {100 * mft/mst}%\nBack: {100 * bck / mst}%\nOptim: {100 * opt / mst}")

def main():
    args = argparser.parse_args()

    world_size = 2
    if args.backend == "gloo":
        model_choice = "toy"
        print("gloo = use toy model")
    else:
        model_choice = args.modelsize
    ddp_type = cls_options[args.ddp]
    param_dict = param_options[model_choice]
    print(f"Using model {model_choice} with params: {param_options[model_choice]=}")
    B = 3 # X and Y each have a batchsize of 2
    DATA = torch.randint(0, param_dict["vocab_size"], (B, param_dict["context_length"]))
    X = DATA[:-1, :]
    Y = DATA[1:, :] # really basic (impossible) targets
    print(f"X,Y has shape {X.shape}, {Y.shape}")
    mp.spawn(fn=distributed_prof, args=(world_size, X, Y, BasicsTransformerLM, args.backend, args.warmupsteps, ddp_type, param_dict), nprocs=world_size, join=True)
    
if __name__ == "__main__":
    main()