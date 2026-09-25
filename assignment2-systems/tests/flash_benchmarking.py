import triton
import torch
import pathlib
import math

from cs336_systems.flash_triton import FlashAttentionTriton

def attn(Q, K, V):
    S: torch.Tensor = Q @ K.mT / math.sqrt(Q.shape[-1])
    mask = ~torch.tril(torch.ones((S.shape[-2], S.shape[-1]), device=Q.device, dtype=torch.bool))
    S = S.masked_fill(mask, -float("inf"))
    P = torch.softmax(S, dim=-1)
    O = P @ V
    return O
    
def pt_forward(Q, K, V):
    return attn(Q, K, V)

def pt_backward(O):
    O.sum().backward(retain_graph=True)

def pt_forward_and_back(Q, K, V):
    O = attn(Q, K, V)
    O.sum().backward()
    
def tr_forward(Q, K, V):
    O = FlashAttentionTriton.apply(Q, K, V, True)

def tr_backward(O):
    O.sum().backward(retain_graph=True)    

def tr_forward_and_back(Q, K, V):
    O = FlashAttentionTriton.apply(Q, K, V, True)
    O.sum().backward()

def benchmark(fp):
    with open(fp, "+w") as f:
        f.write(f"Impl,dtype,batch,seq_len,d_model,mean_time\n")
        B = 1
        Ts = [2**n for n in range(7, 17)]
        Cs = [2**n for n in range(4, 8)]
        for T in Ts:
            for C in Cs:
                for dt in [torch.float32, torch.bfloat16]:
                    O_torch = None
                    torch.cuda.empty_cache()
                    Q = torch.randn((B, T, C), dtype=dt, device="cuda", requires_grad=True)
                    K = torch.randn_like(Q, requires_grad=True)
                    V = torch.randn_like(Q, requires_grad=True)
                    try:
                        O_torch = pt_forward(Q, K, V)
                    except torch.cuda.OutOfMemoryError:
                        pass
                    O_triton = FlashAttentionTriton.apply(Q, K, V, True)
                        
                    p_fwd = lambda: pt_forward(Q, K, V)
                    t_fwd = lambda: tr_forward(Q, K, V)
                    p_bac = lambda: pt_backward(O_torch)
                    t_bac = lambda: tr_backward(O_triton)
                    p_fab = lambda: pt_forward_and_back(Q, K, V)
                    t_fab = lambda: tr_forward_and_back(Q, K, V)
                    try:
                        p_res = triton.testing.do_bench(p_fwd)
                    except torch.cuda.OutOfMemoryError:
                        p_res = "OOM"    
                    f.write(",".join([str(x) for x in ["Pytorch Forward",dt, B, T, C, p_res]]) + "\n")
                    try:
                        t_res = triton.testing.do_bench(t_fwd)
                    except torch.cuda.OutOfMemoryError:
                        t_res = "OOM"    
                    f.write(",".join([str(x) for x in ["Triton Forward",dt, B, T, C, t_res]]) + "\n")
                    try:
                        if O_torch is None:
                            ptb_res = "OOM"
                        else:
                            ptb_res = triton.testing.do_bench(p_bac)    
                    except torch.cuda.OutOfMemoryError:
                        ptb_res = "OOM"
                    f.write(",".join([str(x) for x in ["Pytorch Backward",dt, B, T, C, ptb_res]]) + "\n")
                    try:
                        trb_res = triton.testing.do_bench(t_bac)    
                    except torch.cuda.OutOfMemoryError:
                        trb_res = "OOM"
                    f.write(",".join([str(x) for x in ["Triton Backward",dt, B, T, C, trb_res]]) + "\n")
                    try:
                        pfab_res = triton.testing.do_bench(p_fab)
                    except torch.cuda.OutOfMemoryError:
                        pfab_res = "OOM"                        
                    f.write(",".join([str(x) for x in ["Pytorch F and B",dt, B, T, C, pfab_res]]) + "\n")
                    try:
                        tfab_res = triton.testing.do_bench(t_fab)
                    except torch.cuda.OutOfMemoryError:
                        tfab_res = "OOM"                           
                    f.write(",".join([str(x) for x in ["Triton F and B",dt, B, T, C, tfab_res]]) + "\n")

                
if __name__ == "__main__":
    fp = "results/flash_bench.csv"    
    (pathlib.Path() / "results").mkdir(parents=True, exist_ok=True)
    benchmark(fp)