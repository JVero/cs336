import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math

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
        
    def step(self, closure: Optional[Callable] = None):
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
            
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            gamma, lr, b1, b2, eps = (group.get(k) for k in ["gamma", "lr", "b1", "b2", "eps"])
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
    
def test_lrs():
    lrs = [1e1, 1e2, 1e3]
    for lr in lrs:
        print(f'{6*"="}{lr}{6*"="}')
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)    
        for t in range(10):
            opt.zero_grad()
            loss = (weights**2).mean()
            print(round(loss.cpu().item(), 2))
            loss.backward()
            opt.step()    
