from cs336_basics.transformer import TransformerLM, RotaryPositionalEmbedding
from cs336_basics.training import AdamW, save_checkpoint, load_checkpoint, learning_rate_scheduler, cross_entropy, get_batch, gradient_clipping

import argparse
from cs336_basics.default_configs import configs

import torch
import numpy as np
import subprocess

from datetime import datetime
import pathlib
import json
import os
import time

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

parser.add_argument('--M', type=float, default=1, help="Threshold for gradient clipping")

# Training hyperparameters
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_steps", type=int, default=1) # without specifying, I'm just seeing if the loop is healthy
parser.add_argument("--checkpoint_interval", type=int)
parser.add_argument("--log_interval", type=int, default=25)

# Persistence flags
parser.add_argument("--runs_dir", type=str, default="runs") # Obviously str, but would prefer explicit
parser.add_argument("--label", type=str)

parser.add_argument("--train_dir", type=str, default="data")
parser.add_argument("--train_data", type=str, required=True)
parser.add_argument("--val_dir", type=str, default="data")
parser.add_argument("--val_data", type=str, required=True)
parser.add_argument("--val_num_batches", type=int, default=10)

parser.add_argument("--rng_seed", type=int, default=0)

def save_config_log(config):
    current_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    current_status = subprocess.check_output(['git', 'status', '--porcelain']).decode('utf-8').strip()
    current_patch = subprocess.check_output(['git', 'diff', 'HEAD'])
    curr_time = datetime.now()
    mmdd = curr_time.strftime("%m%d-%H%M%S")
    l = args.label or ""
    if l != "":
        l = f"-{l}"
    lr_label = np.format_float_scientific(args.lr, precision=0, exp_digits=1, trim='-')
    label_prefix = pathlib.Path(args.train_data).with_suffix("")
    label = f"{label_prefix}{l}-lr{lr_label}-{mmdd}".replace("-train","")
    run_dir: pathlib.Path = pathlib.Path(config['runs_dir']) / label
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
    
    run_dir = save_config_log(vars(args))
    
    train_dir = pathlib.Path(args.train_dir)
    train_data: str = args.train_data
    val_dir = pathlib.Path(args.val_dir)
    val_data: str = args.val_data
    
    X_train = np.load(train_dir / train_data, mmap_mode="r")
    X_valid = np.load(val_dir / val_data, mmap_mode="r")
    empirical_vocab_size = 1+np.max(X_train)
    assert lm_vals['vocab_size'] >= empirical_vocab_size, f"{lm_vals['vocab_size']} < {empirical_vocab_size}"
    
    torch.random.manual_seed(args.rng_seed)
    
    model = TransformerLM(**lm_vals)
    optimizer = AdamW(model.parameters(), **optim_vals)
    
    num_steps = args.num_steps
    padding = len(str(num_steps))

    batch_size = args.batch_size
    context_length = args.context_length
    device = args.device
    checkpoint_interval = max(1, args.checkpoint_interval or num_steps // 10) # checkpoint every 10%
    log_interval = args.log_interval
    
    save_checkpoint(model, optimizer, 0, run_dir / "initial.pt")
    load_checkpoint( run_dir / "initial.pt", model, optimizer)
    os.remove(run_dir / "initial.pt")
    
    metrics_headers = ",".join(["step","training_loss","validation_loss","lr","steps per second", "total time"]) + "\n"
    print(metrics_headers, end="", flush=True)
    with open(run_dir / "metrics.csv", "w+") as f:
        f.write(metrics_headers)
    a_max, a_min = args.a_max, args.a_min
    M = args.M
    Tw = round(args.Tw_frac * num_steps)
    prev_time = time.time()
    if device == "mps":
        model.compile(backend="aot_eager")
    else:
        model.compile()
    try:
        start_time = time.time()
        for step in range(num_steps):
            optimizer.zero_grad()
            X, Y = get_batch(X_train, batch_size, context_length, device)
            multiplier = learning_rate_scheduler(step, a_max, a_min, Tw, num_steps)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * multiplier
            y_pred = model(X)
            loss = cross_entropy(y_pred, Y)
            loss.backward()
            gradient_clipping(model.parameters(), M)
            
            # Checkpointing 
            if step % checkpoint_interval == 0 and step != 0:
                save_checkpoint(model, optimizer, step, run_dir / f"ckpt_{step:0{padding}}.pt")
                ckpts = sorted(list(run_dir.glob("ckpt_*.pt")))
                ckpts = ckpts[:-3]
                for ckpt in ckpts:
                    ckpt.unlink()
            # Logging
            if step % log_interval == 0:    
                with torch.no_grad():
                    cur_time = time.time()
                    total_elapsed = cur_time - start_time
                    # Get validation loss
                    total_val_loss = 0
                    elapsed_time = cur_time - prev_time
                    for _ in range(args.val_num_batches):
                        X, Y = get_batch(X_valid, batch_size, context_length, device)
                        y_pred = model(X)
                        valid_loss = cross_entropy(y_pred, Y)
                        total_val_loss += valid_loss.item()
                    avg_val_loss = total_val_loss / args.val_num_batches
                    prev_time = cur_time
                    steps_per_s = log_interval / elapsed_time
                    if step == 0:
                        steps_per_s = 0
                    log_lr = args.lr * multiplier
                    with open(run_dir / "metrics.csv", "a") as f:
                        write_line = [str(round(s, 4)) for s in [step, loss.item(), avg_val_loss, log_lr, steps_per_s, total_elapsed]]
                        write_line[3] = str(round(log_lr, 8)) # round all but the lr
                        log_line = ",".join(write_line)
                        print(log_line)
                        f.write(log_line)
                        f.write("\n")
            optimizer.step()
                        
    except KeyboardInterrupt:
        print("Training stopped early by user. Saving checkpoint...")
        save_checkpoint(model, optimizer, step, run_dir / "killed_run.pt")
    else:
        save_checkpoint(model, optimizer, num_steps, run_dir / "final_checkpoint.pt")