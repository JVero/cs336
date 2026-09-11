# CS336: Language Modeling from Scratch

Self-study workspace for [Stanford CS336](https://stanford-cs336.github.io/) (spring 2026 edition), worked alongside Claude in coach mode: I write every line of the implementations, Claude explains, reviews, debugs by asking questions, and runs the tooling. The rules are in [`CLAUDE.md`](CLAUDE.md).

## Layout

```
assignment1-basics/     BPE tokenizer, Transformer LM, AdamW, training loop   (runs on the Mac)
assignment2-systems/    profiling, FlashAttention-2 in Triton, DDP, FSDP        (needs NVIDIA GPU)
assignment3-scaling/    scaling laws                                            (needs training API or Modal)
assignment4-data/       Common Crawl filtering, dedup, quality classifier       (needs GPU for training)
assignment5-alignment/  GRPO / RL on OLMo-2 1B, optional SFT + DPO supplement   (needs NVIDIA GPU, vLLM)
notes/                  written answers and experiment notes, one file per assignment
scripts/                data download and other tooling
UPSTREAM.md             which upstream commit each starter was copied from
```

Each assignment directory is a self-contained `uv` project copied from the official student repo. The handout PDF sits inside each one.

## Quick start

```sh
# environment (uv manages its own Python; nothing touches the system install)
cd assignment1-basics && uv sync

# run the tests (all fail with NotImplementedError until tests/adapters.py is wired up)
uv run pytest
uv run pytest tests/test_train_bpe.py -x

# data for assignment 1
../scripts/download_a1_data.sh          # TinyStories (~2 GB)
../scripts/download_a1_data.sh --owt    # plus OpenWebText sample (~12 GB unzipped)
```

## Compute

Local machine is an Apple M2 Max with 32 GB and the MPS backend. That covers all of assignment 1, including the small TinyStories training runs. Assignments 2 to 5 depend on Triton, NCCL, flash-attn, and vLLM, so they need an NVIDIA box. The handouts quote budgets in B200 hours. Plan: rent a GPU when starting assignment 2.

## Progress

Ticked when the tests pass or the written answer is in `notes/`.

### Assignment 1: Basics

- [x] `unicode1` Understanding Unicode (1 pt)
- [x] `unicode2` Unicode Encodings (3 pts)
- [x] `train_bpe` BPE Tokenizer Training (15 pts)
- [x] `train_bpe_tinystories` BPE Training on TinyStories (2 pts)
- [x] `train_bpe_expts_owt` BPE Training on OpenWebText (2 pts)
- [x] `tokenizer` Implementing the tokenizer (15 pts)
- [x] `tokenizer_experiments` Experiments with tokenizers (4 pts)
- [x] `linear` Implementing the linear module (1 pt)
- [x] `embedding` Implement the embedding module (1 pt)
- [x] `rmsnorm` Root Mean Square Layer Normalization (1 pt)
- [x] `positionwise_feedforward` Implement the position-wise feed-forward network (2 pts)
- [x] `rope` Implement RoPE (2 pts)
- [x] `softmax` Implement softmax (1 pt)
- [x] `scaled_dot_product_attention` Implement scaled dot-product attention (5 pts)
- [x] `multihead_self_attention` Implement causal multi-head self-attention (5 pts)
- [x] `transformer_block` Implement the Transformer block (3 pts)
- [x] `transformer_lm` Implementing the Transformer LM (3 pts)
- [x] `transformer_accounting` Transformer LM resource accounting (5 pts)
- [x] `cross_entropy` Implement cross-entropy (1 pt)
- [x] `learning_rate_tuning` Tuning the learning rate (1 pt)
- [x] `adamw` Implement AdamW (2 pts)
- [x] `adamw_accounting` Resource accounting for training with AdamW (2 pts)
- [x] `learning_rate_schedule` Implement cosine learning rate schedule with warmup (1 pt)
- [x] `gradient_clipping` Implement gradient clipping (1 pt)
- [x] `data_loading` Implement data loading (2 pts)
- [x] `checkpointing` Implement model checkpointing (1 pt)
- [x] `training_together` Put it together (4 pts)
- [x] `decoding` Decoding (3 pts)
- [x] `experiment_log` Experiment logging (3 pts)
- [x] `learning_rate` Tune the learning rate (2 B200 hrs) (3 pts)
- [x] `batch_size_experiment` Batch size variations (1 B200 hr) (1 pt)
- [x] `generate` Generate text (1 pt)
- [x] `layer_norm_ablation` Remove RMSNorm and train (0.5 B200 hrs) (1 pt)
- [x] `pre_norm_ablation` Implement post-norm and train (0.5 B200 hrs) (1 pt)
- [x] `no_pos_emb` Implement NoPE (0.5 B200 hrs) (1 pt)
- [ ] `swiglu_ablation` SwiGLU vs. SiLU (0.5 B200 hrs) (1 pt)
- [ ] `main_experiment` Experiment on OWT (2 B200 hrs) (2 pts)
- [ ] `leaderboard` Leaderboard (10 B200 hrs) (6 pts)

### Assignment 2: Systems

- [ ] `benchmarking_script` Benchmarking Script (4 pts)
- [ ] `nsys_profile` Nsight Systems Profiling (5 pts)
- [ ] `mixed_precision_accumulation` Mixed-Precision Accumulation (1 pt)
- [ ] `benchmarking_mixed_precision` Benchmarking Mixed Precision (2 pts)
- [ ] `memory_profiling` Memory Profiling (4 pts)
- [ ] `gradient_checkpointing` Memory-Optimal Gradient Checkpointing (4 pts)
- [ ] `torch_compile` Torch Compile (2 pts)
- [ ] `flash_forward` FlashAttention-2 Forward Pass (15 pts)
- [ ] `flash_backward` FlashAttention-2 Backward Pass (5 pts)
- [ ] `flash_benchmarking` FlashAttention-2 Benchmarking (5 pts)
- [ ] `distributed_communication_single_node` Distributed Communication (Single Node) (5 pts)
- [ ] `naive_ddp` Naïve DDP (5 pts)
- [ ] `naive_ddp_benchmarking` Naïve DDP Benchmarking (3 pts)
- [ ] `minimal_ddp_flat_benchmarking` Minimal DDP with Flat Gradients Benchmarking (2 pts)
- [ ] `ddp_overlap_individual_parameters` DDP with Overlapping Individual Parameters (5 pts)
- [ ] `ddp_overlap_individual_parameters_benchmarking` DDP Overlapping Individual Parameters Benchmarking (1 pt)
- [ ] `optimizer_state_sharding` Optimizer State Sharding (15 pts)
- [ ] `fsdp` Fully-Sharded Data Parallel (15 pts)
- [ ] `fsdp_accounting` FSDP Accounting (5 pts)
- [ ] `alternate_ring_all_reduce` Alternate ring all-reduce (1 pt)
- [ ] `data_parallel_calcs` Data parallel calculations (3 pts)
- [ ] `fsdp_calcs` Fully sharded data parallel calculations (3 pts)
- [ ] `tp_calcs` Tensor parallel calculations (4 pts)
- [ ] `fsdp_tp_calcs` 2D parallelism calculations (6 pts)
- [ ] `leaderboard` Leaderboard: fastest training step (10 pts)

### Assignment 3: Scaling

- [ ] `chinchilla_isoflops` IsoFLOPs scaling laws (5 pts)
- [ ] `scaling_laws` Constructing scaling laws leaderboard (50 pts)

### Assignment 4: Data

- [ ] `extract_text` HTML to text conversion (3 pts)
- [ ] `mask_pii` Personally identifiable information (3 pts)
- [ ] `harmful_content` Harmful content (6 pts)
- [ ] `gopher_quality_filters` Gopher quality filters (3 pts)
- [ ] `quality_classifier` Quality classifier (15 pts)
- [ ] `minhash_deduplication` MinHash + LSH document deduplication (8 pts)
- [ ] `filter_data` Filter data for language modeling (6 pts)
- [ ] `inspect_filtered_data` Inspect filtered data (4 pts)
- [ ] `tokenize_data` Tokenize data (2 pts)
- [ ] `train_model` Train model (8 pts)

### Assignment 5: Alignment

- [ ] `baseline_calcs` Compute the variance of the policy gradient estimator (5 pts)
- [ ] `tokenize_prompt_and_output` Prompt and output tokenization (1 pt)
- [ ] `get_response_log_probs` Response log-probs (and entropy) (1 pt)
- [ ] `compute_rollout_rewards` Computing the rewards of rollouts (1 pt)
- [ ] `compute_group_normalized_rewards_grpo` Group normalization (1 pt)
- [ ] `aggregate_loss_across_microbatch_sequence` Aggregate loss across tokens and sequences (0.5 pts)
- [ ] `grpo_train_step_standard_on_policy` GRPO train step (5 pts)
- [ ] `grpo_experiments_standard_on_policy` Use GRPO to improve OLMo-2-0425-1B performance on GSM8K (2 B200 hrs) (10 pts)
- [ ] `grpo_prompt_ablation` Prompt ablation (4 B200 hrs) (3 pts)
- [ ] `think_about_length_normalization` Think about length normalization (1 pt)
- [ ] `compute_group_normalized_rewards_drgrpo` Dr. GRPO Group normalization (0.5 pts)
- [ ] `aggregate_loss_across_microbatch_constant` Dr. GRPO loss aggregation (0.5 pts)
- [ ] `think_about_advantage_normalization` Think about advantage normalization (2 pts)
- [ ] `compute_group_normalized_rewards_maxrl` MaxRL Group normalization (0.5 pts)
- [ ] `grpo_train_step_variants_on_policy` GRPO train step variants (2.5 pts)
- [ ] `grpo_experiments_variants_on_policy` Compare the performance of different RL algorithms (8 B200 hrs) (10 pts)
- [ ] `derive_surrogate_objectives` Derive surrogate objectives for importance reweighting approaches (2 pts)
- [ ] `compute_policy_gradient_loss_off_policy` Off-policy policy gradient with token- level reweighting (1 pt)
- [ ] `think_about_importance_reweighting` Think about importance reweighting (2 pts)
- [ ] `compute_policy_gradient_loss_off_policy_gspo` Off-policy policy gradient with sequence-level reweighting (1 pt)
- [ ] `grpo_train_step_off_policy` Off-policy GRPO train step (2.5 pts)
- [ ] `try_your_own` Try your own policy gradient estimator (10 pts)

#### Supplement: safety, SFT, RLHF (optional)

- [ ] `mmlu_baseline` Zero-shot MMLU baseline (4 pts)
- [ ] `gsm8k_baseline` Zero-shot GSM8K baseline (4 pts)
- [ ] `alpaca_eval_baseline` Zero-shot AlpacaEval baseline (4 pts)
- [ ] `sst_baseline` Zero-shot SimpleSafetyTests baseline (4 pts)
- [ ] `look_at_sft` Inspect instruction tuning data (4 pts)
- [ ] `data_loading` Implement data loading (3 pts)
- [ ] `sft_script` Training script: instruction tuning (4 pts)
- [ ] `sft` Instruction tuning (3 B200 hrs) (6 pts)
- [ ] `mmlu_sft` Evaluate SFT on MMLU (4 pts)
- [ ] `gsm8k_sft` Evaluate SFT on GSM8K (4 pts)
- [ ] `alpaca_eval_sft` Evaluate SFT on AlpacaEval (4 pts)
- [ ] `sst_sft` Evaluate SFT on SimpleSafetyTests (4 pts)
- [ ] `red_teaming` Red-team the instruction-tuned model (4 pts)
- [ ] `look_at_hh` Inspect HH preference data (2 pts)
- [ ] `dpo_loss` DPO loss (2 pts)
- [ ] `dpo_training` DPO training (1 B200 hr) (4 pts)
