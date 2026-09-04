# Assignment 1: Basics

Handout: `assignment1-basics/cs336_assignment1_basics.pdf`

One heading per graded problem. Written answers, experiment numbers, plots, and things I want to remember go here.

## unicode1

Understanding Unicode (1 point)
a. chr(0) returns a string with only a null character
b. on the Python REPL, chr(0) shows as "\x00" but in print is an invisible character.
c. If this character occurs in text, it mainly throws off comparisons. For example "hi" + chr(0) "." != "hi."

## unicode2
a. UTF-8 is the majority of web data, it has a predictible representation, and it takes up less space than UTF-32. 
b. Decoding bytes one at a time is incorrect because there are some characters like e with an accent on top that need to be decoded as a pair rather than something that can be done byte by byte. The correct implementation is bytestring.decode("utf-8") "é".encode("utf-8") is something that breaks in their function
c. A two byte sequence that doesn't encode to any unicode character is 0xFF 0xFF

Unicode Encodings (3 points)

## train_bpe
In train_bpe.py

BPE Tokenizer Training (15 points)
## train_bpe_tinystories
a. I got this done in 74 seconds using about 1.2gb of ram using 12 workers and 120 chunks of the dataset. The longest tokens are whole words the same length as ' accomplishment' that commonly occur in the dataset, which makes sense.
b. The part of the tokenizer training process that takes the most time is finding the next eligible merge target, as profiled when looking at max over the pair counts, it takes 62% of the runtime
BPE Training on TinyStories (2 points)


## train_bpe_expts_owt

BPE Training on OpenWebText (2 points)


## tokenizer

Implementing the tokenizer (15 points)


## tokenizer_experiments

Experiments with tokenizers (4 points)


## linear

Implementing the linear module (1 point)


## embedding

Implement the embedding module (1 point)


## rmsnorm

Root Mean Square Layer Normalization (1 point)


## positionwise_feedforward

Implement the position-wise feed-forward network (2 points)


## rope

Implement RoPE (2 points)


## softmax

Implement softmax (1 point)


## scaled_dot_product_attention

Implement scaled dot-product attention (5 points)


## multihead_self_attention

Implement causal multi-head self-attention (5 points)


## transformer_block

Implement the Transformer block (3 points)


## transformer_lm

Implementing the Transformer LM (3 points)


## transformer_accounting

Transformer LM resource accounting (5 points)


## cross_entropy

Implement cross-entropy (1 point)


## learning_rate_tuning

Tuning the learning rate (1 point)


## adamw

Implement AdamW (2 points)


## adamw_accounting

Resource accounting for training with AdamW (2 points)


## learning_rate_schedule

Implement cosine learning rate schedule with warmup (1 point)


## gradient_clipping

Implement gradient clipping (1 point)


## data_loading

Implement data loading (2 points)


## training_together

Put it together (4 points)


## decoding

Decoding (3 points)


## experiment_log

Experiment logging (3 points)


## learning_rate

Tune the learning rate (2 B200 hrs) (3 points)


## batch_size_experiment

Batch size variations (1 B200 hr) (1 point)


## generate

Generate text (1 point)


## layer_norm_ablation

Remove RMSNorm and train (0.5 B200 hrs) (1 point)


## pre_norm_ablation

Implement post-norm and train (0.5 B200 hrs) (1 point)


## no_pos_emb

Implement NoPE (0.5 B200 hrs) (1 point)


## swiglu_ablation

SwiGLU vs. SiLU (0.5 B200 hrs) (1 point)


## main_experiment

Experiment on OWT (2 B200 hrs) (2 points)


## leaderboard

Leaderboard (10 B200 hrs) (6 points)


