from pathlib import Path
import torch
import json
from cs336_basics.transformer import TransformerLM, RotaryPositionalEmbedding, softmax
from cs336_basics.training import cross_entropy, get_batch
from cs336_basics.tokenizer import Tokenizer

import argparse
import numpy as np

# run_dir = Path("./runs/tinystories-0909-191751")
runs = [str(d) for d in list(Path("./runs").glob("*/"))]


parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str)
parser.add_argument("--p", type=float, default = 0.9)
parser.add_argument("--max_length", type=int, default=300)
parser.add_argument("--temperature", type=float, default=1)

parser.add_argument("--vocab", type=str)
parser.add_argument("--merges", type=str)

parser.add_argument("--run_directory", choices=runs, required=True)

args = parser.parse_args()
tok = Tokenizer.from_files(args.vocab, args.merges)

eps = 1e-8

X = tok.encode(args.input)
X = torch.tensor(X)

end = tok.i_vocab[b"<|endoftext|>"]

run_dir = Path(args.run_directory)
ckpt = run_dir / "final_checkpoint.pt"
params_fname= str(run_dir / "config.json")
if not ckpt.parent.is_dir():
    raise ValueError("Dir doesn't exist?")
if not ckpt.is_file():
    raise ValueError("File doesn't exist?")

with open(params_fname, "r") as f:
    config = json.load(f)

valid_keys = ("d_model", "num_heads", "d_ff", "vocab_size", "context_length", "num_layers", "device")
params = {k: config[k] for k in valid_keys}
params["dtype"] = torch.float32
rope = RotaryPositionalEmbedding(config["theta"], params["d_model"] // params["num_heads"], params["context_length"], device=config["device"])
context_length = params["context_length"]

model = TransformerLM(**params, rope=rope)
state_dict = torch.load(ckpt)

model.load_state_dict(state_dict["model"])

model.eval()

torch.manual_seed(0)
X_cur = X.clone().to(config["device"])
X_out = torch.tensor([], device=config["device"])

with torch.no_grad():
    for i in range(args.max_length):
        logits = model(X_cur[-context_length:])[-1, :].squeeze(dim=0)
        ps = softmax(logits/(args.temperature+eps))
        idx = torch.argsort(ps, descending=True)
        cumsum = torch.cumsum(ps[idx], 0)
        idx_s = cumsum < args.p
        last_pos = idx_s.sum() + 1
        next_tok = torch.multinomial(ps[idx[:last_pos]], num_samples=1)
        
        X_cur = torch.cat([X_cur, idx[next_tok]])
        if idx[next_tok.item()] == end:
            print("<|endoftext|> was emitted")
            break
        print(tok.decode(idx[next_tok].tolist()), end="", flush=True)