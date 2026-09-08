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

if __name__ == "__main__":
    test_lrs()