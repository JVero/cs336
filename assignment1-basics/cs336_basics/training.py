import torch
from cs336_basics.transformer import softmax

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