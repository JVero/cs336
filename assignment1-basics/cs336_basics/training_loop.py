from cs336_basics.transformer import TransformerLM
from cs336_basics.training import AdamW, save_checkpoint, learning_rate_scheduler, cross_entropy

import argparse
from cs336_basics.default_configs import configs

import torch

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
parser.add_argument("--context_length", type=int, default=1024)

parser.add_argument("--device", type=str, choices=available_devices, default=available_devices[0])

# AdamW hyperparameters
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--betas", nargs=2, type=float, default=(0.9, 0.999))
parser.add_argument("--eps", type=float, default=1e-8)

# Training hyperparameters
parser.add_argument("--batch_size", type=int, default=1024)

if __name__ == "__main__":
    # the rest of my script
    args = parser.parse_args()
    lm_args = ("num_layers", "d_model", "num_heads", "vocab_size", "device", "context_length", "d_ff")
    optim_args = ("lr", "weight_decay", "betas", "eps")
    parser.set_defaults(**configs[args.model])
    args = parser.parse_args()
    lm_vals = {k: getattr(args, k) for k in lm_args}

    lm_vals["d_ff"] = lm_vals.get("d_ff") or 64 * round((8 * lm_vals["d_model"] // 3) / 64)    
    
    optim_vals = {k: getattr(args, k) for k in optim_args}
    
    model = TransformerLM(**lm_vals)
    adam = AdamW(model.parameters(), **optim_vals)

