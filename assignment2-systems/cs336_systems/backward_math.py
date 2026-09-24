import torch
import math

@torch.compile
def backward_math(Q, K, V, O, L, grad_out, is_causal=False):
    dO = grad_out
    # 𝐷 = rowsum(𝑶 ∘ 𝒅𝑶)
    D = torch.sum(O * dO, dim=-1)
    # eq 13
    S = Q @ K.mT / math.sqrt(Q.shape[-1])
    if is_causal:
        q_range = torch.arange(Q.shape[-2], device=Q.device)
        k_range = torch.arange(K.shape[-2], device=Q.device)
        kept_mask = q_range[:, None] >= k_range[None, :]
        S[..., ~kept_mask] = -float("inf")
    # eq 14 
    Pij = torch.exp(S - L[..., :, None])
    # eq 15
    dV = Pij.mT @ dO
    # eq 16
    dP = dO @ V.mT
    # eq 17
    dSij = Pij * (dP - D[..., :, None])
    dQ = dSij @ K / math.sqrt(Q.shape[-1])
    dK = dSij.mT @ Q / math.sqrt(Q.shape[-1])
    return dQ, dK, dV

