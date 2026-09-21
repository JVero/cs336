import torch
from torch import nn

from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange
import math

from cs336_basics.nn_utils import softmax, ctx_range
import numpy as np

import argparse
import timeit
from matplotlib import pyplot as plt 

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="mps")
parser.add_argument("--out", default="attn_times.csv")
parser.add_argument("--warmup", default=5, type=int)
parser.add_argument("--num_iters", default=100, type=int)
class BasicAttn(nn.Module):
    """Multi-Head Self-Attention

    This function implements section 3.2.2 of the Transformer paper. In particular,
    given an input tensor of shape `(batch_size, sequence_length, d_model)`, we project
    it to create queries, keys, and values, and then perform causal multi-headed attention with
    those queries, keys, and values.

    Args:
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.

    Returns:
        Tensor of shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(
        self,
        d_model: int,
    ):
        super().__init__()
        self.d_model = d_model

    @ctx_range("Self-attention")
    def forward(
        self, Q, K, V, require_grad=True
    ) -> Float[Tensor, " ... seq d_v"]:
        """
        Args:
            x: The input to perform multi-headed self-attention on.
            positional_ids: The positional indices along the sequence dimension of the input embeddings.

        Returns:
            Self-attention outputs.
        """
        *batch_dims, sequence_length, d_model = Q.size()
        assert d_model == self.d_model

        # Construct causal mask
        iota = torch.arange(sequence_length, device=Q.device)
        qi = rearrange(iota, "query -> query 1")
        kj = rearrange(iota, "key   -> 1   key")
        causal_mask = qi >= kj  # (query, key)
        causal_mask = causal_mask.__getitem__((None,) * len(batch_dims) + (...,))  # Add appropriate leading dimensions

        # Shape: (..., num_heads, sequence_length, d_k)
        QKt: torch.Tensor = Q @ K.mT
        QKt = QKt.masked_fill(~causal_mask, -float('inf'))
        attn_output = softmax(QKt / math.sqrt(self.d_model)) @ V
        
        return attn_output

def sync(device=None):
    torch.accelerator.synchronize()

def fw(Q, K, V, model, device):
    sync(device=device)
    model(Q,K,V)
    sync(device=device)
    
def bw(y: torch.Tensor, device):
    ### Mostly a no-op, but harmless
    sync(device=device)
    # retain_graph so 1. I'm only timing the backward pass, not the freeing, and 2. I don't need to re-allocate y every time
    y.backward(retain_graph=True)
    sync(device=device)  

if __name__ == "__main__":
    args = parser.parse_args()
    batch_size = 8
    results = []
    for d_model in [16, 32, 64, 128]:
        for seq_len in (seqs:=[256, 1024, 4096, 8192, 16384]):
            Q, K, V = [torch.randn((batch_size, seq_len, d_model), device=args.device, requires_grad=True) for _ in range(3)]
            model = BasicAttn(d_model).to(args.device)
            forward = lambda: fw(Q, K, V, model, args.device)
            try:
                ### Forward warmup
                for _ in range(args.warmup):
                    forward()
                ### Forward benchmarking
                result = timeit.repeat(forward, repeat=args.num_iters, number=1)
                with open(args.out, "a+") as f:
                    f.write(f"Forward: {d_model=}, {seq_len=}, {np.mean(result)}\n")
                results.append(np.mean(result))
            except torch.cuda.OutOfMemoryError:
                with open(args.out, "a+") as f:
                    f.write(f"{d_model=}, {seq_len=} went OOM\n")
                ### Forward memory allocation, only available on CUDA, not doing MPS
            if args.device == "cuda":
                print(torch.cuda.memory_allocated())
                with open(args.out, "a+") as f:
                    f.write("Memory after attention: " + str(torch.cuda.memory_allocated()) + "\n")
                    f.write("\n")
            
            try:
                ### Dummy backward variable
                y = model(Q, K, V).sum()
                with open(args.out, "a+") as f:
                    f.write("Memory after y allocated: " + str(torch.cuda.memory_allocated()) +"\n")
                    f.write("\n")
                backward = lambda: bw(y, args.device)
                ### Backward warmup
                for _ in range(args.warmup):
                    backward()    
                with open(args.out, "a+") as f:
                    f.write("Memory after warmup: " + str(torch.cuda.memory_allocated()) + "\n")
                ### Backward benchmarking
                result = timeit.repeat(backward, repeat=args.num_iters, number=1)
                with open(args.out, "a+") as f:
                    f.write(f"Backward: {d_model=}, {seq_len=}, {np.mean(result)}\n")

                results.append(np.mean(result))
                y = None
                with open(args.out, "a+") as f:
                    f.write("Memory after backward profile: " + str(torch.cuda.memory_allocated()) + "\n")
            except torch.cuda.OutOfMemoryError:
                with open(args.out, "a+") as f:
                    f.write(f"{d_model=}, {seq_len=} went OOM\n")
            ### Free y after each iteration, so it doesn't take up unnecessary memory