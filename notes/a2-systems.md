# Assignment 2: Systems

Handout: `assignment2-systems/cs336_assignment2_systems.pdf`

One heading per graded problem. Written answers, experiment numbers, plots, and things I want to remember go here.

## benchmarking_script

Benchmarking Script (4 points)
Using cs336_systems/bench_script.py

First working command `uv run -m cs336_systems.bench_script --vocab_size 32000 --context_length 256 --d_model 768 --num_layers 12 --num_heads 12 --d_ff 2048 --label smoke --data owt_train`

For the below commands ignore the label because there's no file persistence yet.

mps run num_steps = 10, num_loops = 5
`uv run -m cs336_systems.bench_script --vocab_size 10000 --context_length 256 --d_model 768 --num_layers 12 --num_heads 12 --d_ff 2048 --label smoke_100 --data TinyStoriesV2-GPT4-train --device mps --forward --forward_and_back --full_step --num_steps 10`
Forward time, Forward and back, Full step
Means: 0.553,1.687,2.11
Stds: 0.004,0.055,0.022

compiled mps run num_steps = 10, num_loops = 5
`uv run -m cs336_systems.bench_script --vocab_size 10000 --context_length 256 --d_model 768 --num_layers 12 --num_heads 12 --d_ff 2048 --label smoke_100 --data TinyStoriesV2-GPT4-train --device mps --forward --forward_and_back --full_step --num_steps 10 --compile`
Forward time, Forward and back, Full step
Means: 0.546,1.723,2.106
Stds: 0.005,0.09,0.017

cpu run num_steps = 10, num_loops = 5
`uv run -m cs336_systems.bench_script --vocab_size 10000 --context_length 256 --d_model 768 --num_layers 12 --num_heads 12 --d_ff 2048 --label smoke_100 --data TinyStoriesV2-GPT4-train --device cpu --forward --forward_and_back --full_step --num_steps 10`
Forward time, Forward and back, Full step
Means: 2.306,6.042,7.301
Stds: 0.04,0.105,0.17

Across all of these runs, each result is rounded to 3 decimal places. Each model is very very consistent in how long the iterations take across num_steps=10 num_repeats=5 (and a default warmup_steps = 5). For context, the means above are for all 10 iterations, so the mean of each function call is 1/num_steps. Both of the MPS versions are very stable with tiny coefficient of variation for the forward passes (~0.001 for forward, ~0.005 for forward and back, ~0.008 for a full step). There's a slight percieved confound where all values use the same X_train, Y_train, but floating point operations are not dependent on what the inputs are.


C. No warmup steps 
mps run num_steps, num_loops = 5
`uv run -m cs336_systems.bench_script --vocab_size 10000 --context_length 256 --d_model 768 --num_layers 12 --num_heads 12 --d_ff 2048 --label smoke_100 --data TinyStoriesV2-GPT4-train --device mps --forward --forward_and_back --full_step --num_steps 10 --warmup_steps 0`

Forward time, Forward and back, Full step
Means: 0.55,1.633,2.01
Stds: 0.032,0.018,0.042

compiled mps run num_steps, num_loops = 5
`uv run -m cs336_systems.bench_script --vocab_size 10000 --context_length 256 --d_model 768 --num_layers 12 --num_heads 12 --d_ff 2048 --label smoke_100 --data TinyStoriesV2-GPT4-train --device mps --forward --forward_and_back --full_step --compile --num_steps 10 --warmup_steps 0`
Forward time, Forward and back, Full step
Means: 1.201,1.676,2.119
Stds: 1.353,0.041,0.086

Looking at the above values, I notice the first benchmark has a very high standard deviation, so I reran the test and printed the timed results of each run, and noticed that the first value includes the cost of compiling. The below result Running torch.mps.synchronize() after compile does not cause it to sync.
Forward time, per run: [3.91, 0.52, 0.52, 0.53, 0.53]

cpu run num_steps
`uv run -m cs336_systems.bench_script --vocab_size 10000 --context_length 256 --d_model 768 --num_layers 12 --num_heads 12 --d_ff 2048 --label smoke_100 --data TinyStoriesV2-GPT4-train --device cpu --forward --forward_and_back --full_step --num_steps 10 --warmup_steps 0`
Forward time, Forward and back, Full step
Means: 2.302,6.35,7.577
Stds: 0.02,0.2,0.378

compiled cpu run num_steps
`uv run -m cs336_systems.bench_script --vocab_size 10000 --context_length 256 --d_model 768 --num_layers 12 --num_heads 12 --d_ff 2048 --label smoke_100 --data TinyStoriesV2-GPT4-train --device cpu --forward --forward_and_back --full_step --num_steps 10 --warmup_steps 0 --compile`
Forward time, Forward and back, Full step
Means: 4.357,6.883,7.182
Stds: 5.229,2.144,0.182
We observe a similar compile behavior in the CPU: [14.82, 1.73, 1.75, 1.73, 1.76]

1 warmup step (only mps), compiled
`uv run -m cs336_systems.bench_script --vocab_size 10000 --context_length 256 --d_model 768 --num_layers 12 --num_heads 12 --d_ff 2048 --label smoke_100 --data TinyStoriesV2-GPT4-train --device mps --forward --forward_and_back --full_step --compile --num_steps 10 --warmup_steps 1`
Forward time, Forward and back, Full step
Means: 0.532,1.611,2.062
Stds: 0.004,0.021,0.036
[0.54, 0.53, 0.53, 0.53, 0.54]
Even only 1 warmup step is enough of a warmup to be consistent with 5 warmup steps, when compiling. This is consistent with the observation that the overhead from compilation happens in the first step. However, we don't see much gains from compilation on MPS at this batchsize. This is probably due to the fact that the majority of runtime is kernel overhead.


Apparently the above stuff is all wrong, so I need to re-run it with --num_steps 1 num_repeats 10
mps, num_steps=1, num_repeats=5,
`uv run -m cs336_systems.bench_script --vocab_size 10000 --context_length 256 --d_model 768 --num_layers 12 --num_heads 12 --d_ff 2048  --data TinyStoriesV2-GPT4-train --device mps --forward --forward_and_back --full_step --num_steps 1 --num_repeats 10`
Apparently all the runs above were invalid, so I re-ran them all on the cloud. I still come to the same conclusions though.
And (wow) the time of all tasks takes goes up as the model size goes up.



RUN A - 5 Steps of Warmup
`modal run scripts/modal_bench.py --out assignment2-systems/modal_sml_fullrun.csv --flags "--vocab_size 10000 --context_length 512 --forward --forward_and_back --full_step --num_steps 1 --num_repeats 10 --models small medium large"`
label,forward mean,forward std,forward and back mean,forward and back std,full step mean,full step std
small,0.02063,5e-05,0.0615,7e-05,0.07022,0.00023
medium,0.05622,0.00027,0.16971,0.00047,0.19256,0.0006
large,0.12709,0.00122,0.38736,0.00072,0.43772,0.00067
xl,0.343,0.001,1.078,0.001,N/A,N/A

RUN B - 1 Step of warmup
`modal run scripts/modal_bench.py --out assignment2-systems/modal_sml_fullrun.csv --flags "--vocab_size 10000 --context_length 512 --forward --forward_and_back --full_step --num_steps 1 --num_repeats 10 --warmup_steps 1 --models small medium large"`
Separate XL run
`modal run scripts/modal_bench.py --out assignment2-systems/modal_sml_fullrun.csv --flags "--vocab_size 10000 --context_length 512 --forward --forward_and_back --num_steps 1 --num_repeats 10 --warmup_steps 1 --models xl"`
label,forward mean,forward std,forward and back mean,forward and back std,full step mean,full step std
small,0.0214,0.00197,0.06196,0.00029,0.07093,0.0002
medium,0.05818,0.00255,0.17095,0.00052,0.19369,0.0006
large,0.13113,0.01227,0.38731,0.00055,0.43802,0.00084
xl,0.33817,0.00053,1.07668,0.00023, N/A, N/A

RUN C - 0 Steps of warmup
`modal run scripts/modal_bench.py --out assignment2-systems/modal_sml_fullrun.csv --flags "--vocab_size 10000 --context_length 512 --forward --forward_and_back --full_step --num_steps 1 --num_repeats 10 --warmup_steps 0 --models small medium large"`
XL command `modal run scripts/modal_bench.py --out assignment2-systems/modal_sml_fullrun.csv --flags "--vocab_size 10000 --context_length 512 --forward --forward_and_back --num_steps 1 --num_repeats 10 --warmup_steps 0 --models xl"`
label,forward mean,forward std,forward and back mean,forward and back std,full step mean,full step std
small,0.06317,0.12736,0.10228,0.11222,0.07045,0.0001
medium,0.0579,0.00448,0.17188,0.00575,0.19341,0.00017
large,0.13148,0.01741,0.38763,0.00022,0.43863,0.00016
xl,0.37443,0.10995,1.1006,0.07832, N/A, N/A

RUN D - 2 Steps of Warmup
`modal run scripts/modal_bench.py --out assignment2-systems/modal_sml_fullrun.csv --flags "--vocab_size 10000 --context_length 512 --forward --forward_and_back --full_step --num_steps 1 --num_repeats 10 --warmup_steps 2 --models small medium large"`
`modal run scripts/modal_bench.py --out assignment2-systems/modal_sml_fullrun.csv --flags "--vocab_size 10000 --context_length 512 --forward --forward_and_back --num_steps 1 --num_repeats 10 --warmup_steps 2 --models xl"`
label,forward mean,forward std,forward and back mean,forward and back std,full step mean,full step std
small,0.02013,3e-05,0.05871,0.00018,0.0659,0.00015
medium,0.05451,0.0001,0.16178,0.00021,0.18228,0.00164
large,0.12406,0.00526,0.37005,0.00013,0.41059,0.00047
xl,0.3398,0.00015,1.07742,0.00057


b. For Run A with warmup, the small model on average takes 0.02s for a forward pass, 0.06s for forward and backward, and 0.07 for a full step including the optimizer. Assuming the computation time is additive (t(full_forward) = overhead + t(forward), t(full_forward_and_backward) = overhead + t(forward) + t(backward), it is easier to solve for the pure time for backward and for the optimizer than it is for purely forward). For small. back = forward_and_back - forward -> back = 0.06 - 0.02 = 0.04, approximately. Forward + flat overhead is 0.02, and optimizer = t(full_step) - t(forward_and_backwards) = 0.07 - 0.06 -> t(optimizer) = 0.01, approximately, all for small. Note that the question asks for *a* backward pass, so I do not need to calculate it for all model sizes. Those were for small model. The full tables could not have been generated for XL because we could not afford to run the test on a GPU large enough to hold the whole optimizer in vram (including weights, Adam parameters, and activations). The same cause is true for the 10B parameter model in any of the 3 modes. The XL model OOMed. For run D, the CV for small forward is 3e-5/0.02 = 0.0015, forward + back is 0.003, 0.002 for a full step. They are all very consistent.
Without warmup the kernels get constructed in the first evaluation, similar to the previous runs, which were invalid because I ran *too many* samples. There is a measureable difference between 1 and 2 steps of warmup, but I would say the major gain in precision here is from doing 1 step of warmup. Especially if we're being so lazy as to say our n of 1 run sampled 10 times is a sufficient sample. Specificially, why would a second warmup step matter at all, if my theory is that step 1 builds the actual lazily evaluated kernel? I'd bet that there's probably a queue that ends up being built in terms of storing the previous result if there's some kind of allocation that gets built/reused, but that's honestly a guess from someone that doesn't know GPU architectures enough to know why its called a warp.


## nsys_profile

Nsight Systems Profiling (5 points)

Profile Forward and Backward Pass (I chose small and medium) as well as three power-of-two context lengths larger than 128 (256, 512, 1024). Context length of 2048 OOMed for the Medium model on an H100. The template I iterated over the get the profiles is the following command (6 separate runs, this isn't templated in the way its written below)

modal run scripts/modal_nsys.py --gpu H100 --nsys "--pytorch=functions-trace" \
    --flags "--context_length {256,512,1024} --model {small,medium} --forward --forward_and_back --full_step"

Part A: Referencing results/medium_512/medium_512_nvtx_sum.csv
**What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?**
10 runs of Forward took a total of 574359656ns range time and 559187362ms proj time, which is a mean of 0.05743s and 0.05591s per iteration, respectively. For future me, range time is timed from the CPU and proj time is timed with respect to the GPU, informally. Compared to timeit 0.05748s, range is within 0.005%. Proj time, comparatively, is 0.1% different. This disparity is probably due to the kernel call itself.
The standard deviations: 0.00025s (timeit) vs  0.00027s (nsys), each sample is precisely measured between timeit and nsys.
 
Forward times for all models
Size, Context, Timeit, Nsys
Small, 256, 0.02542s, 0.0262432083s,
Small, 512, 0.02207s, 0.02197872437s,
Small, 1024, 0.04535s, 0.04413475547s, 
Medium, 256, 0.03955s, 0.0381409661s,
Medium, 512, 0.05748s, 0.0559187362s,
Medium, 1024, 0.1258s, 0.1240647279s

Part B: 

Q: What CUDA kernel takes the most cumulative GPU time during the forward pass?
The cuda kernel that takes the most time during the forward pass is 
small256:   sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_stage3_warpsize1x4x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas <- this one is different from the other 2
small512:   sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x128x8_stage3_warpsize2x2x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas
small1024:  sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x128x8_stage3_warpsize2x2x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas
medium256:  sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x256x8_stage3_warpsize1x4x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas <- this one is different from the other 2
medium512:  sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x128x8_stage3_warpsize2x2x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas
medium1024: sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x128x8_stage3_warpsize2x2x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas

Q: How many times is this kernel invoked during a single forward pass of your model?
small256: 84 invocations 🤔 🤨
small512: 61 invocations
small1024: 97 invocations
medium256: 48 invocations??? 
medium512: 169 invocations
medium1024: 193 invocations

Q: Is it the same kernel that takes the most runtime when you do both forward and backward passes?
small256:  yes sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_stage3_warpsize1x4x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas
small512:  no void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nt_align1>(T1::Params)
small1024: yes sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x128x8_stage3_warpsize2x2x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas
medium256:  no void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>(T1::Params)
medium512:  yes sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x128x8_stage3_warpsize2x2x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas
medium1024: yes sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x128x8_stage3_warpsize2x2x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas

Part C:
What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?
The elementwise kernels, reduce kernels (Mean and Max) and concatenating which is done in something like RoPE

small256
Total Time, type of kernel
4521565, elementwise mult	
2131294, elementwise mult	
1813504, elementwise add
1362334, elementwise divide
1329311, concatenate array
1182144, max calculation
1173663, elementwise where
1108224, elementwise add
1050943, elementwise copy
973088, mean calculation
In total, these are about 16,600,000ns , out of a total of 90,870,000ns in forward passes, so it takes about 18% of the entire Forward pass.

For each of the rest, I'll report the percent of non-matmul on total time in the forward pass
small512, total 188e6, 47e6 nonmm -> 25% total runtime
small1024, total 422e6, 142e6, -> 33% total runtime
medium256, total 254e6, 36e6, -> 14% total runtime
medium512, total 529e6, 119e6,  -> 22.4% total runtime
medium1024, total 1214e6, 375e6 nonmm -> 30% total runtime

Part D:
**How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)?**
When doing the full step with the optimizer,

Profile running one complete training step with your implementation of AdamW (i.e., the forward pass, computing the loss and running a backward pass, and finally an optimizer step, as you’d do during training). How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?
Looking at

to get full_AdamW
modal run scripts/modal_nsys.py --gpu H100 --label full_AdamW --nsys "--pytorch=functions-trace" \
    --flags "--context_length 512 --model medium --forward --forward_and_back --full_step"

to get forward_only:
modal run scripts/modal_nsys.py --gpu H100 --label forward_only --nsys "--pytorch=functions-trace" \
    --flags "--context_length 512 --model medium --forward"

medium512_full_AdamW_cuda_gpu_kern_sum vs 
medium512_forwardonly_cuda_gpu_kern_sum
sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x128x8_stage3_warpsize2x2x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas is 67.5% of total execution time in forward only, yet only 19.8% of total execution time in a full step, while their raw execution times are within 0.4% of each other (35900544ns forward only vs 35773616ns). For AdamW, 113451005ns of 180973066ns (approximately 63%) of function runtime comes from functions containining the substring "gemm" or "cublas", while for the pure forward, that ratio is 39204593ns vs 53205067ns (approximately 74%). The other kernels are the complement of that, so 37% are non-MatMuls in the Full Step AdamW version, and 26% are non-MatMuls in the pure forward.

This was done for medium 512 only.

Part E:
**Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?**
These numbers are sourced from medium_512_nvtx_kern_sum, inside the :Softmax range, and :Self-Attention range
The runtime in softmax in medium512 is approximately 171ms based on that.
The runtime for the matmuls within a all self-attention layers (Filtered by ":Self-attention" with the final 2 kernel containing gemm or cublas (the first one is the projections which we're not counting here)) is 98ms, for QKt and PV.

Softmax has a much higher ratio of runtime to flops calculated, so it is less efficient.

Flops of self-attn vs softmax -> (8BTCC + 4BTTC) / (5BHTT) = (8CC + 4TC) / (5HT)

For medium 512 (C = 1024, H = 16, T = 512) that is about 256. So 380 ms of matmul against 171 ms of softmax, with 256x the FLOPs.
so the efficiency ratio of matmuls vs softmax is 256/(380/171) -> matmuls are about 115x "denser" than softmax with respect to compute utilization

This happens because softmax requires 5 separate passes over the matrix to do the intermediate calculations, while matmuls can do more simultaneous calculations per step
medium1024 626ms for softmax vs 863ms selfattn -> 1.36x more time
small256 15ms vs - you know what, I'm only doing this for medium with 512 context.. 


## mixed_precision_accumulation

Mixed-Precision Accumulation (1 point)
F32 + F32,       tensor(10.0001) <- 0.001% error
F16 + F16,       tensor(9.9531, dtype=torch.float16) <- ~0.5% error
F32 + F16,       tensor(10.0021) <- 0.02% error
F32 + (F32)F16   tensor(10.0021) <- 0.02% error

The first approach is the most precise, but uses twice the amount of memory for the entire step. The second approach uses the least amount of memory but has about .5% error over 1000 steps, which would compound quickly over long training sessions (the number increments by 9.95 instead of 10, so 0.05/10 = 0.005 = 0.5%).The third and fourth runs have bit-identical results, which implies that `tensor(0, dtype=torch.float32) + tensor(0.01, dtype=torch.float16)` implicitly upcasts the torch.float16 when necessary.

## benchmarking_mixed_precision

Benchmarking Mixed Precision (2 points)

Part A:
• the model parameters within the autocast context? -> Model parameters are used for matmuls AND reductions (optimizer steps), so the latter necessitates *full precision*.
• the output of the first feed-forward layer (ToyModel.fc1)? -> FC1 is a matmul so that would be *lower precision*.
• the output of layer norm (ToyModel.ln)? -> layernorm is a reduction, so it is *full precision*.
• the model’s predicted logits? -> the logits from from self.fc2 which means it is a matmul, which is *lower precision*.
• the loss? -> The loss, using cross-entropy, is a reduce rather than a matmul, so this would be *full precision*.
• the model’s gradients? -> Model gradients are accumulated element-wise, so they would be *full precision*

Part B:
What parts of layernorm would be sensitive to mixed precision? The major risk of using float16 in LayerNorm stems from the fact that calculating the variance requires squaring numbers, which can potentially overflow, due to the reduced expressible range compared to float32. bfloat16 has an increased expressible range such that overflow is less likely than when using float16 (equally likely to float32), but calculating variance is a reduction (E((x - E[x])^2) is an average of squared numbers), which accumulates errors quickly under bfloat16, as is demonstrated in the problem for mixed_precision_accumulation. Actually, mixed_precision_accumulation is a more accurate approximation than bfloat16 would be, as float16 has 3 more bits of mantissa, so bfloat16 would accumulate errors faster.

Part C:
The relevant diff
```python
bench_parser.add_argument("--mixed_precision", action="store_true")

model_params=vars(model_args)
device = model_params.pop("device")

cm = torch.autocast(device, dtype=torch.bfloat16) if bench_args.mixed_precision else nullcontext()
with cm:
    # rest of code
```
The command for both precisions, with their results below:

`modal run scripts/modal_bench.py --gpu H100 --flags "--context_length 512 --model medium --forward --forward_and_back"`
`modal run scripts/modal_bench.py --gpu H100 --flags "--mixed_precision --context_length 512 --model medium --forward --forward_and_back"`
gpu,model_size,context_length,precision,forward mean,forward std, forward and back mean, forward and back std
H100,small,256,full,  0.02542,0.00125,0.07571,0.00717
H100,small,256,mixed, 0.01565,0.00048,0.04093,0.00167

mixed is about 60% of the runtime of the full, 54% for forward and back

H100,small,512,full,  0.02207,0.00067,0.06455,0.00087
H100,small,512,mixed, 0.01559,0.00068,0.04083,0.00119

70% of the runtime for forward, 61% for f and b

H100,small,1024,full, 0.04535,0.00054,0.13749,0.00024
H100,small,1024,mixed,0.02248,0.00055,0.06673,0.00043

50% forward, 49% f and b

H100,medium,256,full,  0.03955,0.00041,0.10255,0.0014
H100,medium,256,mixed, 0.02999,0.00101,0.08295,0.00164

76% of forward, 80% for f and b

H100,medium,512,full,  0.05698,0.00073,0.17052,0.0003
H100,medium,512,mixed, 0.0323,0.00086, 0.08754,0.00236

56% of the forward, 51% f and b

H100,medium,1024,full, 0.1258, 0.00022,0.38289,0.00024
H100,medium,1024,mixed,0.05696,0.00046,0.17122,0.00034

44% of forward, 45% of f and b

H100,large,512,full,0.12241,0.00093,0.37097,8e-05
H100,large,512,mixed,0.04139,0.0008,0.12978,0.00335
33% of forward, 34% of f and b

All of the model sizes get major speedups using bfloat16, and the gains are similar between forward and forward and backward. The gains grow with context length and model size (larger models get more speedup, larger contexts get more speedup)

## memory_profiling

Memory Profiling (4 points)
    # Start recording memory history.
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    ... # what you want to profile in your benchmarking script
    # Save a pickle file to be loaded by PyTorch's online tool.
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    # Stop recording history.
    torch.cuda.memory._record_memory_history(enabled=None)

Running the forward pass and full step for 2048 xl, visualizing the allocations in xl_2048_fp32_forward.jpg and xl_2048_fp32_fullstep.jpg , you can see a pyramid shape forming for the activations with the peak occurring towards the end of `forward`. To the left of `fullstep` you can also see a short decline for the zero_grad() call. I ran all of the runs with batchsize 1, as XL with batchsize of 4 OOMed at full precision at context length 2048 so I wanted to evaluate all of the runs in the same context.

B. Peak memory usage (256, 512, 1024 full steps given as a bonus) is given below. The first row for each context length is full precision, second is mixed precision.

Context Length, Forward,    Full Step
128  Full        14.0GiB,   51.2GiB
     Mixed       19.8GiB,   57.6GiB
256  Full                   51.2GiB
     Mixed                  57.6GiB
512  Full                   51.2GiB
     Mixed                  57.6GiB
1024 Full                   56.2GiB
     Mixed                  57.6GiB
2048 Full       64.1GiB,    90.9GiB
     Mixed      54.9GiB,    82.5GiB


C. This demonstrates the fixed cost of autocast / mixed precision that is "paid for" when the context length grows, which increases the activation size, as naive attention grows T^2 for a (B,T,C) input, and the residual and FFN activations grow linearly with T.

D. Given the reference hyperparameters 
    "xl": {
            "d_model": 2560,
            "d_ff": 10240,
            "num_layers": 32,
            "num_heads": 32
    }, the size of the activations in the residual stream is (B, T, C) = (1, 1024, 2560) = 2,621,440 float32s = 10,485,760 bytes, so 10 MiB. In mixed precision, that count would be halved to 5 MiB.

E: In fp32 at 1024 context length, when you reduce the allocations in the forward pass to only the largest ones, the unanimously largest allocations are 134217728 byte (128 MiB) allocations, which the trace says come from softmax. These are also the largest allocations in mixed precision. They also have the annoying property of persisting longer than most of the other allocations in this filter. The form the biggest sub-pyramid underneath these allocations.

F:
The tiny screenshot is under results/nsys_memory_screenshot. Its unreadableness is why I opted to parse through the sqlite file instead. Rather than the interpreting the screenshot of the memory allocation, I will post the percent of memory allocation for a block (this is block 6 but they are all equivalent)
```josephvero@Mac assignment2-systems % uv run python ../scripts/nsys_alloc_pivot.py results/xl_1024_memory_h100/xl_1024_memory_h100.sqlite --block 6```
block 6: 526.322 ms to 543.013 ms on the timeline ruler (from 'Self-attention' #6 to #7)
in use at window start    2913.1 MiB
in use at window end      3470.4 MiB   (growth 557.4 MiB)
allocated in window       1199.5 MiB in 52 cudaMallocs
still alive at end         557.4 MiB in 22 tensors
op                         n       MiB  % of alive
aten::exp                  1     128.0       23.0%
aten::div                  1     128.0       23.0%
aten::mul                  6     120.0       21.5%
aten::bmm                  3      90.0       16.1%
aten::sigmoid              1      40.0        7.2%

The top 5 contributions are exp, div, mul (element-wise multiplication), batched matmul, and calculating sigmoid. These 5 operations take 90.8% of the memory allocation of each block.

For backwards, when running `uv run python ../scripts/nsys_alloc_pivot.py results/xl_1024_memory_h100/xl_1024_memory_h100.sqlite --range SigmoidBackward0 --block 4`,  (all blocks have equivalent values)

block 4: 1250.106 ms to 1293.621 ms on the timeline ruler (from 'SigmoidBackward0' #4 to #5)
in use at window start   17833.2 MiB
in use at window end     17675.8 MiB   (change -157.4 MiB)
allocated in window       2878.3 MiB in 92 cudaMallocs
freed in window            647.4 MiB allocated before the window, 2388.3 MiB allocated inside it
still alive at end         490.0 MiB in 12 tensors

op                         n       MiB  % of alive
aten::empty_strided        7     400.0       81.6%
aten::mul                  2      80.0       16.3%
aten::bmm                  1      10.0        2.0%
aten::sum                  2       0.0        0.0%

Backwards shrinks total memory usage by 157.4MiB per layer when calculating backward. It roughly matches my expectation because the slope after the peak is more shallow than before the peak (forward allocates more memory than backward). This net comes from the calculation of X - 557.4 MiB = - 157.4, implying that 400MiB of tensors were allocated and not freed within this block. 400MiB in a backwards step would match up if each layer had 400MiB of parameters. 400MiB = 419,430,400 bytes, and if the gradients are stored as float32, then 400MiB would correspond to 104,857,600 floats per layer.
Calculating the number of parameters
4 · 2560²        =  26,214,400 <- Wq, Wk, Wv, Wo
3 · 2560 · 10240 =  78,643,200 <- 3x FFN which are either d_model x d_ffn parameters
total            = 104,857,600 floats = 400 MiB
These match up exactly, and the 7 weight matrices aligns perfectly with the 7 calls to aten::empty_strided.

## gradient_checkpointing
Memory-Optimal Gradient Checkpointing (4 points)
A. The checkpointing strategy that reduces peak memory usage is a recursive tree structure. For a sequence of N=2^n layers, breaking up the layers into checkpoints of length N/2 -> N/(2*2) -> ... 1 has the optimal peak memory usage. For a toy case, [[[1], [2]], [[3], [4]]] where [L] is a checkpoint of a layer. Assume an outer checkpoint takes memory a and a layer's residual takes memory R. This structure I defined has 7 checkpoints. Another example, N=2, [[1], [2]] has 3 checkpoints, by induction, you can see that a model with N=2^n layers, when recursively checkpointed in this manner, will have 2(N-1)+1 = 2N-1 checkpoints, as any 2 equally sized checkpoints of size N/2 can be wrapped by an additional checkpoint. This is an interesting result that I am keeping for my own understanding and visualization, and this disclaimer is here so Claude stops bugging me about it. In the forward pass the peak memory usage is when the final layer is being computed when some tensors are still lingering, and all the checkpoints are stored via checkpoint(layer, x). Inner checkpoints do not save their own inputs or outputs, so only 1 input tensor x0 is stored. This is possible because the input of [1] is the same as the input of [[1],[2]], and every intermediate point inside the inner checkpoint can be recomputed on the fly. At a checkpoint with memory usage a and residual memory size R, the cost is C(N) = a + R, during the forward pass, assuming the top 2 branches are wrapped by a top level checkpoint. The compute cost of **forward** does not change compared to the uncheckpointed version, besides the initial tree traversal, which is trivial compared to the matmuls.

For the description of the backwards pass, let me label the following checkpoints:
[[1], [2]], [[3], [4]] -> C1234 ~<- I understand you said this is unnecessary but let me reason out why rather than take your word for it~
[[1], [2]] -> C12
[[3], [4]] -> C34
[1],[2],[3],[4] -> C1, C2, C3, C4, for consistency
C1234 = C12, C34
When running backwards on C1234, it runs backwards on its last element C34. C34 runs backwards on its last element [4], which requires computing the output of C4. Each of these get put on a stack and then solved.

C1234 -> C34 -> Compute Forward, storing all intermediate tensors a, the final output (for backwards) and no residuals R (3a)
      -> C4  -> Compute backwards, storing then dropping its R, and the output backwards came from (peak 3a + R)
      -> C3  -> Compute backwards, storing then drpoping its R, and the output backwards came from (2a + R)
      -> Drop the intermediates from C34
      -> C12 -> Compute Forward, storing all intermediate tensors a, the final output (for backwards) and no residuals R (3a)
      -> C2 -> Compute backwards, storing then dropping its R, and the output backwards came from (peak 3a + R)
      -> C1 -> Compute backwards, storing then drpoping its R, and the output backwards came from (2a + R)
      -> Drop the intermediates from C12

Now let me try for N=8
[[[1],[2]],[[3],[4]]],[[[5],[6]],[[7],[8]]]
Forward:
Every layer is computed, x0 and x4 are saved
Backward
C5678 Backward (x0 and x4 are here at total cost 2a)
    C5678 Forward storing the inner states x4 and x6 (3a, since x4 is redundant)
    C78 Backward
        C78 Forward (keep x7, keep x6 (4a: x0, x4, x6, x7))
        C8 Backward (Transiently have R for [8], then drop both R and x7 (peak of 4a + R, down to 3a))
        C7 Backward (Transiently have R for [7], then drop R (down to 3a))
    C78 Done, drop x6 (down to 2a)
    C56 Backward: (2a: x0, x4)
        C56 Forward (keep x4, keep x5 (3a))
        C6 Backward (Transiently have R for [6], then drop both R and x5 (down to 2a))
        C5 Backward (Transiently have R for [5], then drop both R and x4 (down to 1a: x0))
C1234 Backward (x0 is here at total cost 1a)
    C1234 Forward storing the inner states x0 and x2 (2a, since x0 is redundant)
    C34 Backward
        C34 Forward (keep x3, keep x2 (3a: x0, x2, x3))
        C4 Backward (Transiently have R for [4], then drop both R and x3 (peak of 3a + R, down to 2a))
        C3 Backward (Transiently have R for [3], then drop R (down to 2a))
    C34 Done, drop x2 (down to 1a)
    C12 Backward: (1a: x0)
        C12 Forward (keep x0, keep x1 (2a))
        C2 Backward (Transiently have R for [2], then drop both R and x1 (down to 1a))
        C1 Backward (Transiently have R for [1], then drop both R and x0 (done))    


The above diagram shows that, when inner checkpoints do not save their inputs, the model must load, then compute, then unload all the values on the fly. Each layer is run n times, where layers are chunked 2^n times (for the above N=8 example, each layer was run forward twice, and once during the backwards pass, so 3 total times. 2^3 = 8, n=3, so as N increases linearly, n increases logarithmically, by definition (N = 2^n, n = log(N)/log(2) = log2(N)), so compute grows O(logN)). Over N layers that scales to O(NlogN). Memory, in this framing, peaks at 4a + R, so peak memory grows at log(N) for a, and it is constant on the scale of R.

Code sample

def ckpt(self, x, layers): # x is first here to disambiguate
    if len(layers) == 1: # base case
        return checkpoint(layers[0], x, use_reentrant=False)
    split = len(layers)//2
    x = checkpoint(self.ckpt, x, layers[:split], use_reentrant=False)
    right = checkpoint(self.ckpt, x, layers[split:], use_reentrant=False)
    return right


Part B:


Solving the equation C = aN/K + Rk , dC/dK = R - aN/K^2, setting that to zero, and solving for K means aN/R = K^2, sqrt(aN/R) = K. Plugging in the appropriate values sqrt((32 * 0.078125)/6.5) ~= 0.62, so a chunk size K of 1 is approximately optimal. Our equation predicts C(0.62) = 32 * 0.078125 / 0.62 + 6.5 * 0.62 = 8.06 , C(1) = 32 * 0.078125 / 1 + 6.5 * 1 = 9, and C(2) = 32 * 0.078125 / 2 + 6.5 * 2 = 14.25, predicting a gap of 14.25 - 9 = 5.25. Measuring this difference is approximately 5.1, so K=1 saves 5.1GiB and takes 0.11s faster. I did not compare against a block smaller than K=1 because I did not do code surgery inside the layers.

Command for K = 1 & K = 2 
`modal run scripts/modal_bench.py --gpu B200 --out assignment2-systems/results/xl_checkpointed_1.csv --flags "--context_length 2048 --model xl --forward_and_back --memory_profiling --checkpoint_chunk_size 1"`
`modal run scripts/modal_bench.py --gpu B200 --out assignment2-systems/results/xl_checkpointed_2.csv --flags "--context_length 2048 --model xl --forward_and_back --memory_profiling --checkpoint_chunk_size 2"`

┌────────────────────────┬───────────┬───────────┐
│                        │    K=1    │    K=2    │
├────────────────────────┼───────────┼───────────┤
│ baseline between steps │ 25.52 GiB │ 25.52 GiB │
├────────────────────────┼───────────┼───────────┤
│ peak                   │ 40.81 GiB │ 45.91 GiB │
├────────────────────────┼───────────┼───────────┤
│ peak above baseline    │ 15.3 GiB  │ 20.4 GiB  │
├────────────────────────┼───────────┼───────────┤
│ time per step          │ 5.01 s    │ 5.12 s    │
└────────────────────────┴───────────┴───────────┘

## pytorch_attention

a. 
Source: `results/attn_B200_20260921-105153.txt`

Command: `python -m cs336_systems.attn_bench --device cuda --out /tmp/attn.txt` (defaults: warmup 5, 100 iters, batch 8)

Times are the mean of 100 passes. Memory is `torch.cuda.memory_allocated()` after `y` is built, so right before backward starts. No config went OOM.

i, ii, iii, iv, v, vi all in the script cs336_systems/attn_bench.py

| d_model | seq_len | forward (ms) | backward (ms) | memory before backward (MiB) |
|--------:|--------:|-------------:|--------------:|-----------------------------:|
| 16  | 256   | 0.173 | 0.840 | 12.6 |
| 16  | 1024  | 0.229 | 0.723 | 82.8 |
| 16  | 4096  | 2.36  | 4.91  | 1062.6 |
| 16  | 8192  | 9.44  | 17.7  | 4189.0 |
| 16  | 16384 | 36.3  | 67.0  | 16681.8 |
| 32  | 256   | 0.166 | 0.831 | 21.1 |
| 32  | 1024  | 0.237 | 1.11  | 84.3 |
| 32  | 4096  | 2.45  | 4.79  | 1068.6 |
| 32  | 8192  | 9.79  | 18.0  | 4201.0 |
| 32  | 16384 | 37.8  | 68.6  | 16705.8 |
| 64  | 256   | 0.185 | 0.800 | 21.8 |
| 64  | 1024  | 0.257 | 1.21  | 87.3 |
| 64  | 4096  | 2.76  | 5.78  | 1080.6 |
| 64  | 8192  | 11.0  | 20.4  | 4225.0 |
| 64  | 16384 | 42.8  | 78.6  | 16753.8 |
| 128 | 256   | 0.189 | 1.32  | 23.3 |
| 128 | 1024  | 0.297 | 1.26  | 93.3 |
| 128 | 4096  | 3.31  | 6.85  | 1104.6 |
| 128 | 8192  | 13.3  | 25.3  | 4273.0 |
| 128 | 16384 | 51.4  | 95.2  | 16849.8 |
Because I used a B200 my models don't get out of memory errors. None of the models run out of memory. I will look at the trends of runtime and memory wrt d_model and seq_len. Runtime goes quadratically with sequence length, and colinearly with s and d (s * d has a coefficient of 0.02, p = 0.011). Memory used during backward grows quadratically with sequence length (coef = 64.6, p < 0.0001), linearly with d_model (coef = 5.7, p < 0.0001 ). The memory complexity for backward has the same factors as forward, which is dominated by sequence length, so a B x 2seq_len x d_model input uses 4x more memory in attention than a B x seq_len x d_model tensor input. What would I do to eliminate this memory cost? Based on the section before this problem, there's a method called FlashAttention-2 that avoids explicitly materializing that seq_len x seq_len attention score matrix.

Memory Accounting, using the 3 major lines in the benchmark script
        QKt: torch.Tensor = Q @ K.mT -> (B, T, C) @ (B, C, T) -> (B, T, T) 
            (B, T, T)= 8 x 16384 x 16384
            2,147,483,648 float32's
            = 8,589,934,592 bytes
            = 8,192 MiB, which isn't directly saved because it's not directly related to the gradients, but I will save this calculation for reference.
        QKt = QKt.masked_fill(~causal_mask, -float('inf'))  <- Save the bytes for the causal mask (T x T) = 268,435,456 bytes = 256 MiB
        attn_output = softmax(QKt / math.sqrt(self.d_model)) @ V
            - softmax(X)
                rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0] <- this operation's gradient doensn't depend on x, so its values aren't saved
                exponentiated_rescaled_input = torch.exp(rescaled_input) -> (B, T, T) = 8,192 MiB
                return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True) -> In-place, this allocates new memory for the bottom value <- allocate a single float32 (4 bytes, trivial)
                exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True) <- This result allocates 8,192 MiB
            - softmax( _ ) @ V
                (B, T, T) @ (B, T, C) = (B, T, C)
                (8, 16384, 16) = 8,388,608 bytes
                = 8 MiB <- Irrelevant! 

        Total:
            0 QKt
          256 causal_mask
        8,192 torch.exp(rescaled_input) = ERI
        8,192 ERI / torch.sum()
           <1 torch.sum()
            8 softmax() @ V
        16648MiB, compared to 16681.8 = 33.8MiB off 


## torch_compile

Torch Compile (2 points)


## flash_forward

FlashAttention-2 Forward Pass (15 points)


## flash_backward

FlashAttention-2 Backward Pass (5 points)


## flash_benchmarking

FlashAttention-2 Benchmarking (5 points)


## distributed_communication_single_node

Distributed Communication (Single Node) (5 points)


## naive_ddp

Naïve DDP (5 points)


## naive_ddp_benchmarking

Naïve DDP Benchmarking (3 points)


## minimal_ddp_flat_benchmarking

Minimal DDP with Flat Gradients Benchmarking (2 points)


## ddp_overlap_individual_parameters

DDP with Overlapping Individual Parameters (5 points)


## ddp_overlap_individual_parameters_benchmarking

DDP Overlapping Individual Parameters Benchmarking (1 point)


## optimizer_state_sharding

Optimizer State Sharding (15 points)


## fsdp

Fully-Sharded Data Parallel (15 points)


## fsdp_accounting

FSDP Accounting (5 points)


## alternate_ring_all_reduce

Alternate ring all-reduce (1 point)


## data_parallel_calcs

Data parallel calculations (3 points)


## fsdp_calcs

Fully sharded data parallel calculations (3 points)


## tp_calcs

Tensor parallel calculations (4 points)


## fsdp_tp_calcs

2D parallelism calculations (6 points)


## leaderboard

Leaderboard: fastest training step (10 points)


