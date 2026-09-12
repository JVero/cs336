from cs336_basics.transformer import TransformerLM
from cs336_basics.training import SGD
import torch

def transformer_accounting():
    vocab_size = 50_257
    context_length = 1_024 # 
    num_layers = 48
    d_model = 1_600
    num_heads = 25
    d_ff = 4_288
    
    n_params = 0    
    
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
    # num_layers (num_layers * d_model + 3 * num_layers * d_model * d_model + d_model * d_model + d_model + 3 * d_model * d_ff) + d_model + d_model * vocab_size

    n_params += num_layers * get_transformer_params()
    
    # RMS Layer
    n_params += d_model
    
    # Linear Layer
    n_params += d_model * vocab_size

    model = TransformerLM(d_model, num_heads, d_ff, vocab_size, context_length, num_layers, device="meta")
    print(sum(p.numel() for p in model.parameters()))

    return n_params # Returns 1640452800

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

def test_lrs():
    lrs = [1e1, 1e2, 1e3]
    for lr in lrs:
        print(f'{6*"="}{lr}{6*"="}')
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)    
        for t in range(10):
            opt.zero_grad()
            loss = (weights**2).mean()
            print(round(loss.cpu().item(), 2))
            loss.backward()
            opt.step()    

def get_activations_count(): # gpt2-xl
    batch_size = 1
    n_layers = 48
    num_heads = 25
    context_length = 1024
    d_model = 1600
    vocab_size = 50257
    
    floats = batch_size * \
    n_layers * 2 * num_heads * context_length * context_length + \
    (1 + n_layers * 56/3) * context_length * d_model + \
    2 * context_length * vocab_size
    print(floats) # 4089153536.0
    n_bytes = 4 * floats
    print(n_bytes) # 16356614144.0
