from cs336_basics.transformer import TransformerLM, RotaryPositionalEmbedding
from cs336_basics.training import AdamW, save_checkpoint, load_checkpoint, learning_rate_scheduler, cross_entropy, get_batch

import argparse
from cs336_basics.default_configs import configs

import torch
import numpy as np
import subprocess

from datetime import datetime
import pathlib
import json
import os

parser = argparse.ArgumentParser()

preset_choices = configs.keys()
default_model = min(configs, key=lambda k: configs[k]["num_layers"])

available_devices = []
if torch.mps.is_available():
    available_devices.append("mps")
if torch.cuda.is_available():
    available_devices.append("cuda")
if torch.cpu.is_available():
    available_devices.append("cpu")

if len(available_devices) == 0:
    raise ValueError("No compatible PyTorch devices")

parser.add_argument("--model", help="Model presets", choices=preset_choices, default=default_model)

# Model Parameters
parser.add_argument("--num_layers", help="Number of transformer layers", type=int)
parser.add_argument("--d_model", help="Dimensionality of the model", type=int)
parser.add_argument("--num_heads", type=int)

parser.add_argument("--vocab_size", type=int, default= 50_257)
parser.add_argument("--d_ff", type=int)
parser.add_argument("--context_length", type=int, default=256)

parser.add_argument("--theta", type=int, default=10_000)

parser.add_argument("--device", type=str, choices=available_devices, default=available_devices[0])

# AdamW hyperparameters
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--betas", nargs=2, type=float, default=(0.9, 0.999))
parser.add_argument("--eps", type=float, default=1e-8)

# Scheduler hyperparameters
parser.add_argument('--a_max', type=float, default=1)
parser.add_argument('--a_min', type=float, default=0.1)
parser.add_argument('--Tw_frac', type=float, default=0.01, help="Fraction of the run warmup")

# Training hyperparameters
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_steps", type=int, default=1) # without specifying, I'm just seeing if the loop is healthy
parser.add_argument("--checkpoint_interval", type=int)

# Persistence flags
parser.add_argument("--runs_dir", type=str, default="runs") # Obviously str, but would prefer explicit
parser.add_argument("--label", type=str)

parser.add_argument("--train_dir", type=str, default="data")
parser.add_argument("--train_data", type=str, required=True)
parser.add_argument("--val_data", type=str)

parser.add_argument("--rng_seed", type=int, default=0)

def save_config_log(config):
    current_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    current_status = subprocess.check_output(['git', 'status', '--porcelain']).decode('utf-8').strip()
    current_patch = subprocess.check_output(['git', 'diff', 'HEAD'])
    curr_time = datetime.now()
    mmdd = curr_time.strftime("-%m%d-%H%M%S")
    run_dir: pathlib.Path = pathlib.Path(config['runs_dir']) / (config['label'] + mmdd)
    run_dir.mkdir(parents=True)
    with open(run_dir / "diff.patch", "wb") as f:
        f.write(current_patch)
        
    config["time"] = curr_time.isoformat()
    config["git"] = {}
    config["git"]["hash"] = current_hash
    config["git"]["status"] = current_status
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    return run_dir
    
if __name__ == "__main__":
    # the rest of my script
    args = parser.parse_args()
    
    label = args.label or args.model or "run"
    
    lm_args = ("num_layers", "d_model", "num_heads", "vocab_size", "device", "context_length", "d_ff")
    optim_args = ("lr", "weight_decay", "betas", "eps")
    parser.set_defaults(**configs[args.model])
    args = parser.parse_args()
    lm_vals = {k: getattr(args, k) for k in lm_args}
    
    rope = RotaryPositionalEmbedding(args.theta, args.d_model // args.num_heads, args.context_length, args.device)
    lm_vals['rope'] = rope
    optim_vals = {k: getattr(args, k) for k in optim_args}

    lm_vals["d_ff"] = lm_vals.get("d_ff") or 64 * round((8 * lm_vals["d_model"] // 3) / 64)    
    args.d_ff = lm_vals["d_ff"]
    args.label = label
    
    run_dir = save_config_log(vars(args))
    
    train_dir = pathlib.Path(args.train_dir)
    train_data: str = args.train_data
    val_data: str | None = args.val_data
    
    X_train = np.load(train_dir / train_data, mmap_mode="r")
    
    empirical_vocab_size = 1+np.max(X_train)
    assert lm_vals['vocab_size'] >= empirical_vocab_size, f"{lm_vals['vocab_size']} < {empirical_vocab_size}"
    
    torch.random.manual_seed(args.rng_seed)
    
    model = TransformerLM(**lm_vals)
    optimizer = AdamW(model.parameters(), **optim_vals)
    
    num_steps = args.num_steps
    batch_size = args.batch_size
    context_length = args.context_length
    device = args.device
    checkpoint_interval = max(1, args.checkpoint_interval or num_steps // 10) # checkpoint every 10%
    
    
    save_checkpoint(model, optimizer, 1, run_dir / "initial.pt")
    load_checkpoint( run_dir / "initial.pt", model, optimizer)
    os.remove(run_dir / "initial.pt")

    a_max, a_min= args.a_max, args.a_min
    Tw = round(args.Tw_frac * num_steps)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        X, Y = get_batch(X_train, batch_size, context_length, device)
        multiplier = learning_rate_scheduler(step, a_max, a_min, Tw, num_steps)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * multiplier
        y_pred = model(X)
        loss = cross_entropy(y_pred, Y)
        loss.backward()
        optimizer.step()