import argparse
import json
import itertools
from copy import deepcopy
import subprocess

import sys

sys.path.append("/scripts")

from scripts.selfhost import run_experiments
parser = argparse.ArgumentParser()

# parser.add_argument("--model_width", nargs="+", type=int)
# parser.add_argument("--model_depth", nargs="+", type=int)

width_depth = {
    "3M": (256, 4),
    "6M": (512, 2),
    "12M": (512, 4),
    "25M": (512, 8),
    "100M": (1024, 8),
}

n_params = {
    "3M": 3.15e6,
    "6M": 6.29e6,
    "12M": 12.58e6,
    "25M": 25.17e6,
    "100M": 100.66e6
}

def token_budget(budget, params_str, batch, n_evals):
    toks = budget / (6*n_params[params_str]) # C = 6DN -> N = C/(6D)
    # round to the multiple of 512 * batch * n_evals
    return (512*batch*n_evals) * round(toks // (512*batch*n_evals))

parser.add_argument("--N", nargs="+", default=["3M"], type=str)
parser.add_argument("--lr", nargs="+", type=float, default=[0.002])
# parser.add_argument("--train_tokens_mult", nargs="+", type=int, default=[1], choices=[1,2,4,8])
parser.add_argument("--batch_size", nargs="+", type=int,default=[32], choices=[32, 64, 128]) # Not worth batch_size 256 because it won't scale to larger models
parser.add_argument("--gpu", default="h100", choices=["h100", "b200"])
parser.add_argument("--num_heads", nargs="+", default=[8], type=int)
parser.add_argument("--max_runtime", type=int, default=120)
parser.add_argument("--budget", type=float, nargs="+", required=True)

args_to_dict = {
    "model_depth": ("architecture_config", "num_hidden_layers"),
    "model_width": ("architecture_config", "hidden_size"),
    "d_ff": ("architecture_config", "intermediate_size"),
    "head_dim": ("architecture_config", "head_dim"),
    "num_heads": ("architecture_config", "num_attention_heads"),
    "lr": ("optimizer_config", "lr_scheduler", "peak_value"),
    "batch_size": ("train_batch_size",),
    "max_runtime": ("max_runtime_seconds",),
    "n_tokens": ("total_train_tokens",),
    "n_evals": ("n_evals"),
}

def set_num_heads(d, num_heads):
    d["architecture_config"]["num_attention_heads"] = num_heads
    d["architecture_config"]["num_key_value_heads"] = num_heads

def assign_nested(d, keys, value):
    inner = d
    for k in keys[:-1]:
        inner = inner[k]
    inner[keys[-1]] = value

def get_nested(d, keys):
    inner = d
    for k in keys:
        inner = inner[k]
    return inner   

json_template = """  {
    "architecture_config": {
      "attention_bias": false,
      "head_dim": 64,
      "hidden_size": 448,
      "intermediate_size": 1280,
      "num_attention_heads": 7,
      "num_hidden_layers": 9,
      "num_key_value_heads": 7,
      "rms_norm_eps": 1e-6,
      "rope_theta": 1000000,
      "tie_word_embeddings": false,
      "dtype": "bfloat16",
      "vocab_size": 32000
    },
    "optimizer_config": {
      "lr_scheduler": {"peak_value": 1e-4, "final_lr_frac": 0.1, "warmup_frac": 0.05, "init_value": 0.0},
      "weight_decay": 0.01,
      "beta1": 0.9,
      "beta2": 0.95,
      "eps": 1e-8,
      "eps_root": 1e-8,
      "grad_clip_norm": 1.0
    },
    "train_batch_size": 128,
    "val_batch_size": 32,
    "n_evals": 16,
    "total_train_tokens": 1048576,
    "max_runtime_seconds": 120,
    "model_seed": 0
  }"""

if __name__ == "__main__":
    args = parser.parse_args()
    model_params_template = json.loads(json_template)
    pairs = [width_depth[n] for n in args.N]
    batch_sizes = args.batch_size or [get_nested(model_params_template, args_to_dict["batch_size"])]
    lrs = args.lr or [get_nested(model_params_template, args_to_dict["lr"])]
    # train_tokens_mults = args.train_tokens_mult
    num_headss = args.num_heads
    all_configs = []
    for lr, batch_size, num_heads, N, budget in itertools.product(lrs, batch_sizes, num_headss, args.N, args.budget):
        # derived value
        pair = width_depth[N]
        width = pair[0]
        depth = pair[1]
        d_ff = 128 * round((8 * width // 3) / 128)
        model_params = deepcopy(model_params_template)
        n_tokens = token_budget(budget, N,batch_size, model_params["n_evals"])

        assign_nested(model_params, args_to_dict["n_tokens"], n_tokens)
        assign_nested(model_params, args_to_dict["model_depth"], depth)
        assign_nested(model_params, args_to_dict["model_width"], width)
        assign_nested(model_params, args_to_dict["d_ff"], d_ff)
        set_num_heads(model_params, num_heads)
        assert get_nested(model_params, args_to_dict["model_width"]) % get_nested(model_params, args_to_dict["num_heads"]) == 0, "Model width must be divisible by num_heads"
        assign_nested(model_params, args_to_dict["head_dim"], width // get_nested(model_params, args_to_dict["num_heads"]))
        assign_nested(model_params, args_to_dict["lr"], lr)
        assign_nested(model_params, args_to_dict["batch_size"], batch_size)
        assign_nested(model_params, args_to_dict["max_runtime"], args.max_runtime)
        # model_params["total_train_tokens"] *= train_tokens_mult
        with open("/tmp/config.json", "w") as f:
            json.dump(model_params, f)
        all_configs.append(model_params)
    run_experiments(all_configs, gpu=args.gpu)