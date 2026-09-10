#!/bin/bash

vals=(
    3e-4
    1e-3
    3e-3
    1e-2
    3e-2
)
    
for item in "${vals[@]}"; do
    uv run -m cs336_basics.training_loop --model tinystories --batch_size 32 --lr $item --label ts-lr-$item --vocab_size 10000 --num_steps 5000 --train_data TinyStoriesV2-GPT4-train.npy --val_data TinyStoriesV2-GPT4-valid.npy
done
