from torch import nn
import torch

import math

from einops import einsum, rearrange

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
                
        # Intermediate calculations
        var = 2 / (self.in_features + self.out_features)
        std = math.sqrt(var)
        
        # Actual weights w/ corresponding initializations
        self.W = nn.Parameter(torch.zeros((out_features, in_features), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.W, mean=0, std=std,a=-3*std, b=3*std)
            
    def forward(self, X):
        return einsum(X, self.W, "... in_features, out_features in_features -> ... out_features")

class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings: int, embedding_dim: int,
                 device = None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
                
        # Intermediate calculations
        self.W = nn.Parameter(torch.zeros((num_embeddings, embedding_dim), dtype=dtype, device=device))
        nn.init.trunc_normal_(self.W, mean=0, std=1,a=-3, b=3)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
    
    @staticmethod
    def rms(a, eps):
        ai2 = einsum(a, a, "... a, ... a ->... a").sum(-1)
        d_model = a.shape[-1]
        rval = torch.sqrt(eps + ai2/d_model)
        return rval
    
    def forward(self, X):
        # X.shape (batch_size, sequence_length, d_model)
        # RMSNorm(ai) = ai * gi / (RMS(a))
        # RMS(a) = sqrt(eps + (1/dmodel) * sum_i^dmodel ai^2)
        
        original_dt = X.dtype
        X = X.to(torch.float32)
        RMS = RMSNorm.rms(X, self.eps)
        X = X * self.g 
        X = einsum(X, 1/RMS, "... seq_len d_model, ... seq_len -> ... seq_len d_model")
        
        return X.to(original_dt)


class SwiGLU(nn.Module):
    # X.shape = (B, d_model)
    # W1, W3 .shape = d_ff, d_model
    # W2 .shape = d_model, d_ff
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        self.d_ff = d_ff or 64 * round((8 * d_model // 3) / 64)
        
        self.W1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.W3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.W2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)

    @staticmethod
    def silu(X):
        return X * torch.sigmoid(X)
    
    def forward(self, X):
        # 𝑊2 (SiLU(𝑊1 𝑥) ⊙ 𝑊3 𝑥)
        # T1        T2      T3  
        
        W1x = self.W1(X)
        SILU = SwiGLU.silu(W1x)
        W3x = self.W3(X)
        inp = SILU * W3x
        return self.W2(inp)
    
    
    
class RotaryPositionalEmbedding(nn.Module):

    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.max_seq_len = max_seq_len

        i = torch.arange(max_seq_len) # max_seq_len
        denominator = theta ** ((2 * torch.arange(1, 1+d_k//2) - 2)/d_k) # d_k / 2
        thetas = i[:, None] / denominator[None, :] # (max_seq_len, d_k // 2)
        Ri = torch.stack([torch.cos(thetas), -torch.sin(thetas), torch.sin(thetas), torch.cos(thetas)], dim=-1)

        R = Ri.view((max_seq_len, d_k//2, 2, 2)).to(device)
        
        self.R = nn.Buffer(R, persistent=False)
        
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x (..., seq_len, d_k)
        # token_positions (..., seq_len)
        x = rearrange(x, "... seq_len (n_pairs r) -> ... seq_len n_pairs r", r=2)
        x = einsum(self.R[token_positions], x, "... seq_len n_pairs r c, ... seq_len n_pairs c-> ... seq_len n_pairs r")
        x = rearrange(x, "... seq_len n_pairs r -> ... seq_len (n_pairs r)")
        return x
    
def softmax(v: torch.Tensor, dim=-1) -> torch.Tensor:
    # V has arbitrary dims
    v_max = torch.amax(v, dim=dim, keepdim=True)
    v = v - v_max
    ev = torch.exp(v)
    return ev / ev.sum(dim=dim, keepdim=True)