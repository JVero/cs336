%%writefile flash_triton.py
import torch

import triton.language as tl
import triton
import math

# TODO (tomorrow):
# x O_block_ptr: batch offset uses stride_qb, should use stride_ob
# x L_block_ptr: add batch offset to base pointer (stride_lb)
# x L_block_ptr: 1D, so strides / offsets / block_shape / order get one entry each
# x Line 6: init accumulators before the loop (O tile, l, m) with tl.zeros / tl.full
# - Loop body: load K, V tiles (boundary_check), then steps 9-13 as in the torch version
# - After loop: step 15 (normalize O, compute L), store O and L (boundary_check)
# - FlashAttentionTriton: allocate O and L, launch grid (Tq, batch), save for backward
# - Test: first stage = copy Q into O to check pointers, then compare to torch version

@triton.jit
def flash_fwd_kernel(Q_ptr, K_ptr, V_ptr,
                     O_ptr, L_ptr,
                     stride_qb, stride_qq, stride_qd,
                     stride_kb, stride_kk, stride_kd,
                     stride_vb, stride_vk, stride_vd,
                     stride_ob, stride_oq, stride_od,
                     stride_lb, stride_lq,
                     N_QUERIES, N_KEYS,
                     scale,
                     D: tl.constexpr,
                     Q_TILE_SIZE: tl.constexpr,
                     K_TILE_SIZE: tl.constexpr,
                     is_causal: tl.constexpr
                    ):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr( 
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0,0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr( # .shape = (Tk, d_model)
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0,0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    # Part of line 6
    O_block_ptr = tl.make_block_ptr( # .shape = (Tq, d_model)
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D,),
        strides=(stride_oq, stride_od,),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D), # (B or 1, Bq, D)?
        order=(1,0)
    )
    L_block_ptr = tl.make_block_ptr( # .shape = (T,)
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), dtype=tl.float32, value=-float("inf"))

    # num_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    num_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    # Line 5
    Qtile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    # still need to initialize l and m
    for i in tl.range(num_k_tiles):
        Ktile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vtile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # Line 9
        Sij = tl.dot(Qtile, tl.trans(Ktile)) * scale
        ## Causal Masking here
        # Line 10
        new_maxes = tl.max(Sij, axis=-1)
        prev_maxes = m
        m = tl.maximum(new_maxes, prev_maxes)
        
        # Line 11
        Pij = tl.exp(Sij - m[:, None]) # <- different from pytorch because m has no batch dim here
        # Line 12 
        rowsumPij = tl.sum(Pij, axis=-1)
        inner = tl.exp(prev_maxes - m)
        l = inner * l + rowsumPij
        
        # Line 13
        Oi = inner[:, None] * Oi + tl.dot(Pij, Vtile)
        
        # The kernel should only have a single loop, which will iterate key tiles 1 ≤ 𝑗 ≤ 𝑇𝑘.
        # Advance block pointers at the end of the loop.
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    # Line 15
    Oi = Oi / l[:, None]
    Li = m + tl.log(l)
    
    tl.store(O_block_ptr,Oi)
    tl.store(L_block_ptr,Li)
    
    
    
class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, Bq=16, Bk=32):
        O = torch.empty_like(Q, device="cuda") # I bet empty_like allocates the device to be Q's device, but just to be explicit
        L = torch.zeros((Q.shape[:-1]), device="cuda")
        N_QUERIES = Q.shape[-2]
        N_KEYS = K.shape[-2]
        scale = 1/math.sqrt(Q.shape[-1])
        flash_fwd_kernel[(triton.cdiv(Q.shape[-2], Bq), Q.shape[0])] (
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS, 
            scale,
            Q.shape[-1],
            tl.constexpr(Bq), tl.constexpr(Bk),
            is_causal=tl.constexpr(is_causal)
        )
        ctx.is_causal = True
        ctx.save_for_backward(Q,K,V,O,L)
        return O