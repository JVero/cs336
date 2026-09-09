configs = {
    "tinystories": {
        "num_layers": 4,
        "d_model": 512,
        "num_heads": 16,
        "vocab_size": 10000,
        "context_length": 256,
        "d_ff": 1344,
        "theta": 10_000,
        "total_tokens": 327_680_000
    },
    "gpt2-small": {
        "num_layers": 12,
        "d_model":  768,
        "num_heads": 12
    },
    "gpt2-medium" : {
        "num_layers": 24,
        "d_model":  1024,
        "num_heads": 16
    },
    "gpt2-large": {
        "num_layers": 36,
        "d_model":  1280,
        "num_heads": 20
        },
    "gpt2-xl":{
        "num_layers": 48,
        "d_model":  1600,
        "num_heads": 25,
    }
}