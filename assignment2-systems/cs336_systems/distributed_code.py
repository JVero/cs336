import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pathlib
import timeit
import time
import numpy as np

import argparse

argp = argparse.ArgumentParser()
worker_choices = [2,4,6]
argp.add_argument("--num_workers", type=int, default=4, choices=worker_choices)
size_choices = ["1MB", "10MB", "100MB", "1GB"]
argp.add_argument("--size", type=str, choices=size_choices, default="1MB")
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
    
def distributed_demo(rank, world_size, n_floats, fp):
    setup(rank, world_size)
    data = torch.ones(n_floats)
    for _ in range(2):
        dist.all_reduce(data)
    dist.barrier()
    pre = time.time()
    n_iters = 10
    for _ in range(n_iters):
        dist.all_reduce(data)
    times = [0 for _ in range(world_size)]
    elapsed = (time.time() - pre)/n_iters
    dist.all_gather_object(times, elapsed)
    if rank!=0 or fp is None:
        return
    with open(fp, "a+") as f:
        f.write(",".join([str(x) for x in [world_size, data.numel() * 4, np.mean(times)]]) + "\n")


def main():
    args = argp.parse_args()
    fdir = pathlib.Path() / "results"
    fdir.mkdir(parents=True, exist_ok=True)
    fp = fdir / "dist_bench.csv"
    with open(fp, "w+") as f:
        f.write("world_size,size_choice,time" + "\n")
        f.flush()
        for world_size in worker_choices:
            for size_choice in size_choices:
                print(f"{world_size=}, {size_choice=}")
                n_floats = num_floats[size_choice]
                ## Warmup
                n_warmup = 0
                res = mp.spawn(fn=distributed_demo, args=(world_size, n_floats, fp), nprocs=world_size, join=True)
                # f.write(",".join([str(x) for x in [world_size, size_choice,res]])+"\n")
            
if __name__ == "__main__":
    main()