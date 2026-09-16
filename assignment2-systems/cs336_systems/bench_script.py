import argparse
from pathlib import Path
import timeit
import gc

import numpy as np
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy, ctx_range
from cs336_basics.optimizer import AdamW

model_parser = argparse.ArgumentParser()
bench_parser = argparse.ArgumentParser()

valid_types = {
    'f32': torch.float32,
    'b16': torch.bfloat16,
    'f16': torch.float16
}

configs = {
    "small":
        {
            "d_model": 768,
            "d_ff": 3072,
            "num_layers": 12,
            "num_heads": 12
        },
    "medium": {
            "d_model": 1024,
            "d_ff": 4096,
            "num_layers": 24,
            "num_heads": 16
    },
    "large": {
            "d_model": 1280,
            "d_ff": 5120,
            "num_layers": 36,
            "num_heads": 20
    },
    "xl": {
            "d_model": 2560,
            "d_ff": 10240,
            "num_layers": 32,
            "num_heads": 32
    },
    "10B": {
            "d_model": 4608,
            "d_ff": 12288,
            "num_layers": 50,
            "num_heads": 36
    }
}

available_devices = []
if torch.cuda.is_available():
    available_devices.append("cuda")
if torch.mps.is_available():
    available_devices.append("mps")
if torch.cpu.is_available():
    available_devices.append("cpu")

params_group = model_parser.add_argument_group("Model parameters")
params_group.add_argument("--vocab_size", type=int, default=10000)
params_group.add_argument("--context_length", type=int, required=True)
params_group.add_argument("--d_model", type=int)
params_group.add_argument("--num_layers", type=int)
params_group.add_argument("--num_heads", type=int)
params_group.add_argument("--d_ff", type=int)
params_group.add_argument("--rope_theta", type=int, default=10_000)
# params_group.add_argument("--dtype", type=str, choices=valid_types.keys(), default='f32')
params_group.add_argument("--device", type=str, choices=available_devices, required=True)

bench_parser.add_argument("--label")
bench_parser.add_argument("--batch_size", type=int, default=4)
bench_parser.add_argument("--num_steps", type=int, default=1)
bench_parser.add_argument("--num_repeats", type=int, default=10)
bench_parser.add_argument("--compile", action="store_true")
bench_parser.add_argument("--forward", action="store_true")
bench_parser.add_argument("--forward_and_back", action="store_true")
bench_parser.add_argument("--full_step", action="store_true")
bench_parser.add_argument("--warmup_steps", type=int, default=5)
bench_parser.add_argument("--model",  help="The max size the run will go until", choices=list(configs.keys()), required=True)

bench_parser.add_argument("--of_name", required=True)

def sync(device=None):
    
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    elif device == "cpu":
        pass # its a no-op
    else:
        if torch.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda" 
        sync(device=device)
    
@ctx_range("Forward")
def forward(model, X, device):
    y = model(X)
    sync(device)

@ctx_range("Forward and Back")
def forward_and_backward(model, X, Y, loss_fn, device):
    y_pred = model(X)
    loss = loss_fn(y_pred, Y)
    loss.backward()
    sync(device)

@ctx_range("Full Step")
def full_step(model, X, Y, loss_fn, optimizer: torch.optim.Optimizer, device):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, Y)
    loss.backward()
    optimizer.step()
    sync(device)
    
def get_rnd_batch(batch_size, context_length, vocab_size, device):
    X = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    Y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    return X, Y

def clear_cache(device):
    if device == "mps":
        torch.mps.empty_cache() # the runs happen multiple times so release the vram
    elif device =="cuda":
        torch.cuda.empty_cache()
    elif device == "cpu":
        pass # no cache to empty

def run_profile(label: str, overridden_args: dict, model_args, bench_args):
    gc.collect()
    clear_cache(model_args.device)
    
    model_params=vars(model_args)
    device = model_params.pop("device")
    
    for k, v in overridden_args.items():
        model_params[k] = v
    
    basics = BasicsTransformerLM(**model_params)
    basics.to(device)
    model_params["device"] = device # Re-assign it 
    if bench_args.compile:
        if device == "mps":
            basics.compile(backend="aot_eager")
        else:
            basics.compile()
    
    optimizer = AdamW(basics.parameters())
    X_sample, Y_sample = get_rnd_batch(bench_args.batch_size, model_params["context_length"], model_params["vocab_size"], device)
    
    f = lambda: forward(basics, X_sample, device)
    fandb = lambda: forward_and_backward(basics, X_sample, Y_sample, loss_fn, device)
    fstep = lambda: full_step(basics, X_sample, Y_sample, loss_fn, optimizer, device)
    loss_fn = cross_entropy
    if bench_args.warmup_steps > 0:
        print("Warming up...")
        with ctx_range("Warmup"):
            for _ in range(bench_args.warmup_steps):
                if bench_args.forward:
                    f()
                if bench_args.forward_and_back:
                    fandb()
                if bench_args.full_step:
                    fstep()
    ### Mac doens't have enough VRAM for this
    if label in ["large", "xl", "10b"] and device == "mps":
        bench_args.full_step = None
    print(f"Number of steps: {bench_args.num_steps}")
    results = []
    with ctx_range("Measurement"):
        if bench_args.forward:
            print("Running forward...")
            result = timeit.repeat(f, number=bench_args.num_steps, repeat=bench_args.num_repeats)
            results.extend([f(result) for f in [np.mean, np.std]])
        if bench_args.forward_and_back:
            print("Running forward and backward...")
            result = timeit.repeat(fandb, number=bench_args.num_steps, repeat=bench_args.num_repeats)
            results.extend([f(result) for f in [np.mean, np.std]])
        if bench_args.full_step:
            print("Running full training step...")
            result = timeit.repeat(fstep, number=bench_args.num_steps, repeat=bench_args.num_repeats)
            results.extend([f(result) for f in [np.mean, np.std]])
    with open(bench_args.of_name, "a") as fo:
        fo.write(label +","+ ",".join([str(round(r, 5)) for r in results])+"\n")
    return results

if __name__ == "__main__":
    model_args, rest = model_parser.parse_known_args()
    bench_args = bench_parser.parse_args(rest)
    tests_to_run = []
    if bench_args.forward:
        tests_to_run.extend(["forward" + " " + s for s in ["mean", "std"]])
    if bench_args.forward_and_back:
        tests_to_run.extend(["forward and back" +  " " + s for s in ["mean", "std"]])
    if bench_args.full_step:
        tests_to_run.extend(["full step" + " " + s for s in ["mean", "std"]])
    header = "label,"  +",".join(tests_to_run) + "\n"
    with open(bench_args.of_name, "w+") as f:
        f.write(header)
    
    k, v = (bench_args.model, configs[bench_args.model])
    try:
        run_profile(k, v, model_args, bench_args)
    except Exception as E:
        print(E)
        print(k, " went OOM")
        clear_cache(model_args.device)