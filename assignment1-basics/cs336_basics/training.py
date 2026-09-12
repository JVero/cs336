import torch
from collections.abc import Callable, Iterable
import typing
import math
import os
from pathlib import Path
import io

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):

    largest = torch.amax(logits, dim=-1, keepdim=True)
    normalized_logits = logits - largest
    m = logits.shape[-2]
    d = logits.shape[0] if len(logits.shape) == 3 else 1 # dataset D
    targets = targets.unsqueeze(-1)
    selected_logits = torch.gather(normalized_logits, -1, targets)
    expoi = torch.exp(normalized_logits).sum(-1, keepdim=True)
    v = 1/(d * m) * (torch.log(expoi) - selected_logits)
    return v.sum()

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        
    def step(self, closure: Callable | None = None): # type: ignore
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss

 


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, weight_decay=0.01, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        # LR is the same as alpha
        if lr < 0:
            raise ValueError(f"Invalid learning rate {lr}")
        defaults = {"gamma": weight_decay,
                    "lr": lr, 
                    "b1": betas[0], 
                    "b2": betas[1], 
                    "eps": eps}
        super().__init__(params, defaults)
            
    def step(self, closure: Callable | None = None): # type: ignore
        loss = None if closure is None else closure()
        for group in self.param_groups:
            gamma, lr, b1, b2, eps = (group[k] for k in ["gamma", "lr", "b1", "b2", "eps"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                g = p.grad.data # Line 6
                at = lr * math.sqrt(1-b2 ** t) / (1 - b1**t) # Line 7
                p.data -= lr * gamma * p.data # Line 8 - Weight Decay
                state["m"] = b1 * state.get("m", torch.zeros_like(p.data)) + (1 - b1) * g # Lines 2 and 9
                state["v"] = b2 * state.get("v", torch.zeros_like(p.data)) + (1 - b2) * g*g # Lines 3 and 10
                p.data -= at * state["m"]/(eps + torch.sqrt(state["v"])) # Line 11
                state["t"] = t+1 # Incrementing, has to do with line 4
        return loss

def learning_rate_scheduler(t, a_max, a_min, Tw, Tc):
    # warmup
    if t < Tw:
        a_t = a_max * t / Tw
    elif t <= Tc:
        frac = (t - Tw)/(Tc - Tw)
        a_t = a_min + 1/2 * (1+math.cos(frac * math.pi)) * (a_max - a_min)
    else:
        a_t = a_min
        
    return a_t

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], M: float, eps=1e-6):
    parameters = list(parameters)
    data = torch.cat([p.grad.flatten() for p in parameters if p.grad is not None])
    l2 = data.norm(2)
    frac = M / (l2 + eps)
    if frac > 1:
        return
    for param in parameters:
        if param.grad is not None:
            param.grad *=  frac

def get_batch(arr, batch_size, context_length: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, len(arr) - context_length, [batch_size], dtype=torch.long)
    offsets = torch.arange(context_length, dtype=torch.long)
    ix = idx[:, None] + offsets[None, :]
    x, y = arr[ix], arr[ix+1]
    
    return torch.as_tensor(x, device=device, dtype=torch.long), torch.as_tensor(y, device=device, dtype=torch.long)


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, iteration: int,
                    out: str| os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    d = {"model": model.state_dict(),
         "optimizer": optimizer.state_dict(), "iteration": iteration}
    if isinstance(out, (str, os.PathLike)): # Named file
        out_tmp = Path(out).with_name(Path(out).name + ".tmp")
        try:
            torch.save(d, out_tmp)
        except Exception:
            out_tmp.unlink(missing_ok = True)
            raise
        os.replace(out_tmp, out)
    elif isinstance(out, (io.IOBase)): # File object handle
        torch.save(d, out)
    else: # Unsupported thing
        raise TypeError(type(out).__name__)
        

def load_checkpoint(src:  str| os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    d = torch.load(src, map_location="cpu")
    model.load_state_dict(d["model"])
    optimizer.load_state_dict(d["optimizer"])
    return d["iteration"]
    