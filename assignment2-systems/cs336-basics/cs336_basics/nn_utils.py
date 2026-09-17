import torch
import contextlib

import torch.cuda.nvtx as nvtx

class MacCTX(contextlib.ContextDecorator):
    def __init__(self, description):
        pass
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass

ctx_range = nvtx.range if torch.cuda.is_available() else MacCTX

@ctx_range("Softmax")
def softmax(x, dim=-1):
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)

@contextlib.contextmanager
def memory_snapshot(fname, clean_cache=False, max_entries=1_000_000):
    torch.cuda.synchronize()
    if clean_cache:
        torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    try:
        yield
    finally:
        torch.cuda.synchronize()
        torch.cuda.memory._dump_snapshot(fname)
        torch.cuda.memory._record_memory_history(enabled=None)


def log_softmax(x, dim=-1):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x = x - x_max
    return x - torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))


def cross_entropy(inputs, targets):
    negative_log_softmax_logits = -log_softmax(inputs)
    return torch.mean(torch.gather(negative_log_softmax_logits, -1, targets.unsqueeze(-1)))


def clip_gradient(parameters, max_norm):
    grads = [p.grad for p in parameters if p.grad is not None]
    norm = torch.tensor(0.0, device=grads[0].device)

    for g in grads:
        norm += (g**2).sum()

    norm = torch.sqrt(norm)
    clip_coef = min(1, max_norm / (norm + 1e-6))
    for g in grads:
        g *= clip_coef
