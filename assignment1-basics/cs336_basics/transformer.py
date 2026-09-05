from torch import nn
import torch

import math
import einops

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
        return einops.einsum(X, self.W, "... in_features, out_features in_features -> ... out_features")

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