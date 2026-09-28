import os
import torch
import torch.distributed as dist

import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from tests.common import ToyModel

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

backend_options = ["gloo", "nccl"]
argparser = argparse.ArgumentParser()
argparser.add_argument("--backend", choices=backend_options, default="gloo")
argparser.add_argument("--modelsize", choices=param_options.keys(), default="xl")

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

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
        
def distributed_prof(rank, world_size, X_full: torch.Tensor, Y_full: torch.Tensor,
                     model_class: type[torch.nn.Module], backend, init_dict=None):
    setup(rank, world_size, backend)
    assert X_full.numel() % world_size == 0, "Batch should be divisible by the total number of workers"
    X_chunk = torch.chunk(X_full, world_size)[rank]
    Y_chunk = torch.chunk(Y_full, world_size)[rank]
    
    if isinstance(init_dict, dict):
        model = NaiveDistributor(model_class(**init_dict))
    else:
        model = NaiveDistributor(model_class())

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
    for _ in range(3):
        y = model(X_chunk)
        loss_val = loss(y, Y_chunk)
        loss_val.backward()
        optim.step()
        optim.zero_grad()
    
    # Clears the previous gradient values, but keeps the graphs
    model.sync_weights()
    
    fwd_times = []
    bck_times = []
    full_step_times = []
    grad_sync_times = []
    opt_times = []
    n_steps = 10
    for _ in range(n_steps):
        dist.barrier()
        step_start = time.perf_counter_ns()
        # Forward time
        fwd_start = time.perf_counter_ns()
        y_pred = model(X_chunk)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fwd_end = time.perf_counter_ns()
        fwd_times.append(torch.Tensor([fwd_end-fwd_start]))
        
        bck_start = time.perf_counter_ns()
        loss_val = loss(y_pred, Y_chunk)
        loss_val.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()        
        bck_end = time.perf_counter_ns()
        bck_times.append(torch.Tensor([bck_end-bck_start]))

        grad_sync_start = time.perf_counter_ns()
        model.ddp_on_after_backward(optim)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        grad_sync_end = time.perf_counter_ns()
        grad_sync_dur = torch.Tensor([grad_sync_end-grad_sync_start])
        grad_sync_times.append(grad_sync_dur)

        opt_start = time.perf_counter_ns()
        optim.step()
        optim.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        opt_end = time.perf_counter_ns()
        opt_times.append(torch.Tensor([opt_end-opt_start]))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_end = time.perf_counter_ns()
        full_step_times.append(torch.Tensor([step_end - step_start]))
    fwd_times = torch.cat(fwd_times).to(device)
    dist.all_reduce(fwd_times)
    if rank == 0:
        mft = (fwd_times / (dist.get_world_size() * n_steps)).mean()
        print(f"Mean forward time: {mft}")
    
    grad_sync_times = torch.cat(grad_sync_times).to(device)
    dist.all_reduce(grad_sync_times) 
    mean_grad_time = grad_sync_times 
    if rank == 0:
        mgt = mean_grad_time.mean()
        print(f"Mean time averaging gradients: {mgt}")

    bck_times = torch.cat(bck_times).to(device)
    dist.all_reduce(bck_times)
    if rank == 0:
        bck_mean = bck_times.mean()
        print(f"Mean backward time: {bck_mean}")

    opt_times = torch.cat(opt_times).to(device)
    dist.all_reduce(opt_times)
    if rank == 0:
        opt_mean = opt_times.mean()
        print(f"Mean optimizer step time: {opt_mean}")

    full_step_times = torch.cat(full_step_times).to(device)
    dist.all_reduce(full_step_times)
    mean_step_time = full_step_times.mean()
    if rank == 0:
        print(f"Mean step time: {mean_step_time}")
    
    if rank == 0:
        mst = mean_step_time
        print(f"Percent of full step: \nGrad: {100 * mgt / mst}%\nFwd: {100 * mft/mst}%\nBack: {100 * bck_mean / mst}%\nOptim: {100 * opt_mean / mst}")

def hide():
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
    pass # this is to hide this commented out code

def main():
    args = argparser.parse_args()

    world_size = 2
    if args.backend == "gloo":
        model_choice = "toy"
        print("gloo = use toy model")
    else:
        model_choice = args.modelsize
    param_dict = param_options[model_choice]
    print(f"Using model {model_choice} with params: {param_options[model_choice]=}")
    B = 3 # X and Y each have a batchsize of 2
    DATA = torch.randint(0, param_dict["vocab_size"], (B, param_dict["context_length"]))
    X = DATA[:-1, :]
    Y = DATA[1:, :] # really basic (impossible) targets
    print(f"X,Y has shape {X.shape}, {Y.shape}")
    mp.spawn(fn=distributed_prof, args=(world_size, X, Y, BasicsTransformerLM, args.backend, param_dict), nprocs=world_size, join=True)
    # res = mp.spawn(fn=, args=(world_size, X, Y, ToyModel, loss_fn, queue,"cpu", None), nprocs=world_size, join=True)

    # torch.random.manual_seed(0)
    # serial_model = BasicsTransformerLM(**param_dict)
    # optim = torch.optim.AdamW(serial_model.parameters())
    # times = []
    # for _ in range(10):
    #     start = time.perf_counter_ns()
    #     y_pred = serial_model(X)
    #     end = time.perf_counter_ns()
    #     times.append(end-start)
    # print(f"Mean serial forward: {np.mean(times)}")
    # loss = cross_entropy(y_pred, Y)
    # loss.backward()
    # optim.step()
    # optim.zero_grad()
    
    # for p1, p2 in zip(mp_model.parameters(), serial_model.parameters()):
    #     if not torch.allclose(p1, p2):
    #         print((p1 - p2).abs().max())
    #         print("Not equal")
    #         return
    # print("Checks out")
    
    
if __name__ == "__main__":
    main()