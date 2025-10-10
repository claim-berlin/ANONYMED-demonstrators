#!/bin/bash
weights=dit.safetensors
num_steps=1
mkdir -p output

for file in "$1"/*
do
    outfile="output/${file##*/}"
    echo "$file -> $outfile"
    python3 mlx_inference.py --load $weights --input $file --output $outfile --num_sample_steps $num_steps
done
