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


b. For Run A with warmup, the small model on average takes 0.02s for a forward pass, 0.06s for forward and backward, and 0.07 for a full step including the optimizer. Assuming the computation time is additive (t(full_forward) = overhead + t(forward), t(full_forward_and_backward) = overhead + t(forward) + t(backward), it is easier to solve for the pure time for backward and for the optimizer than it is for purely forward). For small. back = forward_and_back - forward -> back = 0.06 - 0.02 = 0.04, approximately. Forward + flat overhead is 0.02, and optimizer = t(full_step) - t(forward_and_backwards) = 0.07 - 0.06 -> t(optimizer) = 0.01, approximately, all for small. Note that the question asks for *a* backward pass, so I do not need to calculate it for all model sizes. Those were for small model. The full tables could not have been generated for XL because we could not afford to run the test on a GPU large enough to hold the whole optimizer in vram (including weights, Adam parameters, and activations). The same cause is true for the 10B parameter model in any of the 3 modes. The XL model OOMed. The CV for small forward is 0.02/3e-5 = 0.0015, forward + back is 0.003, 0.002 for a full step. They are all very consistent.
Without warmup the kernels get constructed in the first evaluation, similar to the previous runs, which were invalid because I ran *too many* samples. There is a measureable difference between 1 and 2 steps of warmup, but I would say the major gain in precision here is from doing 1 step of warmup. Especially if we're being so lazy as to say our n of 1 run sampled 10 times is a sufficient sample. Specificially, why would a second warmup step matter at all, if my theory is that step 1 builds the actual lazily evaluated kernel? I'd bet that there's probably a queue that ends up being built in terms of storing the previous result if there's some kind of allocation that gets built/reused, but that's honestly a guess from someone that doesn't know GPU architectures enough to know why its called a warp.


## nsys_profile

Nsight Systems Profiling (5 points)


## mixed_precision_accumulation

Mixed-Precision Accumulation (1 point)


## benchmarking_mixed_precision

Benchmarking Mixed Precision (2 points)


## memory_profiling

Memory Profiling (4 points)


## gradient_checkpointing

Memory-Optimal Gradient Checkpointing (4 points)


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


