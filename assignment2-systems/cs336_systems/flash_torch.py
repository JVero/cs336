import torch
from einops import rearrange
import math
from cs336_systems.backward_math import backward_math

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False):
        # Line 1
        Bq, Bk = 16, 32 # <- I assume its nice when these match, but maybe they don't need to, somehow?
        *B, _, d_model = Q.shape

        num_q_tiles = math.ceil(Q.shape[-2] / Bq)
        num_k_tiles = math.ceil(K.shape[-2] / Bk)
        q_tiles, k_tiles, v_tiles = [], [], []
        Otiles = []
        Ltiles = []
        # Line 2
        for i in range(num_q_tiles):
            q_tiles.append(Q[...,i*Bq:(i+1)*Bq,:])
        # Line 3
        for i in range(num_k_tiles):
            k_tiles.append(K[...,i*Bk:(i+1)*Bk,:])
            v_tiles.append(V[...,i*Bk:(i+1)*Bk,:])
        # Line 4
        for i in range(num_q_tiles):
            # Line 5
            Qtile = q_tiles[i] # Explicitly doing this, to match the algorithm
            # Line 6
            Oi = torch.zeros((*B, Qtile.shape[-2], d_model), device=Q.device)
            lij = torch.zeros(Qtile.shape[-2], device=Q.device)
            mij = -float('inf') * torch.ones(Qtile.shape[-2], device=Q.device)
            # Line 7
            for j in range(num_k_tiles):
                # Line 8
                Ktile = k_tiles[j]
                Vtile = v_tiles[j]
                # Line 9
                Sij = Qtile @ Ktile.mT / math.sqrt(d_model)
                # Line 10
                new_maxes = torch.max(Sij, dim=-1).values
                mijm1 = mij
                mij = torch.max(mij, new_maxes)
                # Line 11
                Pij = torch.exp(Sij - mij[..., :, None])
                # Line 12 
                rowsumPij = torch.sum(Pij, dim=-1)
                inner = torch.exp(mijm1 - mij)
                lij = inner * lij + rowsumPij
                # Line 13
                Oi = inner[..., :, None] * Oi + Pij @ Vtile
            # Line 14, end the for loop <- obvious, but just so *every* line is accounted for
            # Line 15
            Oi = Oi / lij[..., None]
            Li = mij + torch.log(lij)
            
            Otiles.append(Oi)
            Ltiles.append(Li)
        O = torch.cat(Otiles,dim=-2)
        L = torch.cat(Ltiles,dim=-1)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal # type: ignore
        return O
    
    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_out):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        return *backward_math(Q, K, V, O, L, grad_out, is_causal=is_causal), None
        
if __name__ == "__main__":
    B = 10
    T = 100
    d_model = 11
    device = "mps" if torch.mps.is_available() else "cuda"
    Q = torch.randn((B, T, d_model), requires_grad=True, device=device)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    ctx = torch.autograd.function.FunctionCtx()
    O = FlashAttentionPytorch.apply(Q, K, V)
    O.sum().backward()