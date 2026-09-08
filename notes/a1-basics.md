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
a. The longest token in the vocab, found using
from cs336_basics.tokenizer import Tokenizer
tok = Tokenizer.from_files("owt_train_vocab.json", "owt_train_merges.json")
max(tok.id_to_bytes.values(), key=len).decode("utf-8")

"ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ" is the result, which is mojibake

b. TinyStories is cleaner because it is 1. shorter, and 2. more curated so there's only real words in it.

## tokenizer

Implementing the tokenizer (15 points)


## tokenizer_experiments

Experiments with tokenizers (4 points)
a. 
Compression ratio for tinystories-train using tinystories-train 4.164006604292791
Compression ratio for owt-train using owt-train 4.690217391304348
b.
Compression ratio for owt-train using tinystories-train 3.1927807219278073. The chunks get smaller because the efficient representation in one domain doesn't necessarily transfer to the other domain
c. When I measured the throughput of 50mb it took 42 seconds. At about 1.2mb/s, it would take about 8 nonstop days to tokenize 825gb of data. Afterwards I built a parallel and cached version of the tokenizer that processed all the data in 314s, which means the speed increased to about 48mb/s on the whole ~15gb dataset.
d. uint16 is an appropriate datatype because for a vocab size of 32_000, you'll need bits such that 2^bits > 32_000 -> log2(32000) = 14.9. uint16 is the smallest value that fits that 

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
```python
def transformer_accounting():
    vocab_size = 50_257
    context_length = 1_024 # 
    num_layers = 48
    d_model = 1_600
    num_heads = 25
    d_ff = 4_288
    
    n_params = 0
    # Embedding
    n_params += vocab_size * d_model
    
    
    # Transformer Layers
    def get_transformer_params():
        n_params = 0
        # rms1
        n_params += d_model
        
        # FMHSA
        n_params += 3 * d_model * d_model
        n_params += d_model * d_model
        
        # rms2
        n_params += d_model
        
        # SwiGLU
        n_params += 3 * d_model * d_ff
        return n_params
    
    n_params += num_layers * get_transformer_params()
    
    # RMS Layer
    n_params += d_model
    
    # Linear Layer
    n_params += d_model * vocab_size

    return n_params # Returns 1640452800

```
1640452800 x 4 bytes = 6.56gb of RAM

Part B
def calculate_flops(vocab_size=50_257,
                    context_length=1_024,
                    num_layers = 48,
                    d_model = 1_600,
                    num_heads = 25,
                    d_ff = 4_288):
    n_flops = 0
    result = {}
    # (context_length, d_model)
    
    # Transformer Layers
    def get_transformer_flops():
        result = {}
        n_flops = 0
        # Wq, Wk, Wv
        result["Wqkv"] = 3 * (2 * context_length * d_model * d_model)
        n_flops += result["Wqkv"]
        
        result["QKt"] = 2 * context_length * d_model * context_length
        # Q Kt (B, T, C) (B, C, T)
        n_flops += result["QKt"]
        
        result["QKt V"] = 2 * context_length * context_length * d_model
        # QKt V (T, T) (T, C) -> (T, C)
        n_flops += result["QKt V"]
        
        result["(QKt V) Wo"] = 2 * context_length * d_model * d_model
        # (QKt V) Wo (T, C) (C, C) -> (T, C)
        n_flops += result["(QKt V) Wo"]
        
        # Linear x 3
        result["Linear"] = 3 * 2 * context_length * d_model * d_ff
        n_flops += result["Linear"]
        return n_flops, result
    
    flops, result['transformer_per_layer'] = get_transformer_flops()
    # (T, C)
    n_flops += num_layers * flops
    
    result["FinalLinear"] = 2 * context_length * d_model * vocab_size
    # final linear (T, C) (C, vocab_size)
    n_flops += result["FinalLinear"]
    
    return n_flops, result

Based on plotting the flops per layer, the Position-Wise Linear layers take up the most flops, in this configuration

    # Part C
    num_layers = 48
    flops, result = calculate_flops(num_layers=num_layers)
    transformer = result.pop("transformer_per_layer")
    for key in transformer:
        transformer[key] *= num_layers 
    bars = result | transformer
    total = sum(bars.values())
    for b in bars:
        bars[b] = round(bars[b] * 100/total, 2)
    
    print(f"num_layers: {num_layers} : {flops}", bars) # 3_516_769_894_400
    plt.bar(bars.keys(), bars.values())
    plt.title("Part C")
    plt.show()
    
    # Part D - GPT-2 Small
    num_layers = 12
    d_model = 768
    num_heads = 12
    flops, result = calculate_flops(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=2048)
    transformer = result.pop("transformer_per_layer")
    for key in transformer:
        transformer[key] *= num_layers
    bars = result | transformer
    total = sum(bars.values())
    for b in bars:
        bars[b] = round(bars[b] * 100/total, 2)
    print(f"GPT-2 Small", flops, bars) # 291_648_307_200
    
    plt.bar(bars.keys(), bars.values())
    plt.title("GPT-2 Small")
    plt.show()
    
    # Part D - GPT-2 Medium
    num_layers = 24
    d_model = 1024
    num_heads = 16
    flops, result = calculate_flops(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=2752)
    transformer = result.pop("transformer_per_layer")
    for key in transformer:
        transformer[key] *= num_layers 
    bars = result | transformer
    total = sum(bars.values())
    for b in bars:
        bars[b] = round(bars[b] * 100/total, 2)
    
    print("GPT-2 Medium ", flops, bars) # 830_172_299_264
    plt.bar(bars.keys(), bars.values())
    plt.title("GPT-Medium")
    plt.show()
    
    # Part D - GPT-2 large
    num_layers = 36
    d_model = 1280
    num_heads = 20
    flops, result = calculate_flops(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=3392)
    transformer = result.pop("transformer_per_layer")
    for key in transformer:
        transformer[key] *= num_layers 
    bars = result | transformer
    total = sum(bars.values())
    for b in bars:
        bars[b] = round(bars[b] * 100/total, 2)
    
    print("GPT-2 Large" , flops, bars) # 131_745_710_080
    plt.bar(bars.keys(), bars.values())
    plt.title("GPT-2 Large")
    plt.show()
    
    # Part E
    num_layers = 48
    context_length = 16_384
    flops, result = calculate_flops(num_layers=num_layers,context_length=context_length)
    transformer = result.pop("transformer_per_layer")
    print("Part E", flops)
    for key in transformer:
        transformer[key] *= num_layers 
    bars = result | transformer
    
    total = sum(bars.values())
    for b in bars:
        bars[b] = round(bars[b] * 100/total, 2)
    
    print("Part E" , flops, bars) # 133_577_729_638_400
    plt.bar(bars.keys(), bars.values())
    plt.title("GPT-2 XL w/ long context")
    plt.show()
    

    Below the total flops are reported, followed by the percentage each part takes up
    GPT-2 Small 291648307200 {'FinalLinear': 27.1, 'Wqkv': 14.91, 'QKt': 6.63, 'QKt V': 6.63, '(QKt V) Wo': 4.97, 'Linear': 39.76}
    
    GPT-2 Medium  830172299264 {'FinalLinear': 12.7, 'Wqkv': 18.62, 'QKt': 6.21, 'QKt V': 6.21, '(QKt V) Wo': 6.21, 'Linear': 50.05}

    GPT-2 Large 1768530903040 {'FinalLinear': 7.45, 'Wqkv': 20.49, 'QKt': 5.46, 'QKt V': 5.46, '(QKt V) Wo': 6.83, 'Linear': 54.3}

    GPT-2 XL : 3516769894400 {'FinalLinear': 4.68, 'Wqkv': 21.47, 'QKt': 4.58, 'QKt V': 4.58, '(QKt V) Wo': 7.16, 'Linear': 57.53}

    Part E 133577729638400 {'FinalLinear': 1.97, 'Wqkv': 9.04, 'QKt': 30.87, 'QKt V': 30.87, '(QKt V) Wo': 3.01, 'Linear': 24.24}    

    D. As the model size increases the proportion of the model dedicated to transformer compared to the final linear increases, and the projection to vocab takes up less of the total FLOPS. As the d_model increases, the output projection from Wo will take more of the relative flops as it grows quadratically with d_model. As the model size increases, the Final Linear projection's flop share decreases and the internal share of linear in the transformer block's flops increase.

    E. The FLOPs get more dominated by the actual attention, because QKt V scales as a function of context_length^2

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


