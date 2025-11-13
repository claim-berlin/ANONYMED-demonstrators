#!/bin/bash
weights=dit.safetensors
num_steps=1
mkdir -p output_CTA

for file in "$1"/*
do
    outfile="output_CTA/${file##*/}"
    echo "$file -> $outfile"
    python3 xmodality/mlx_inference.py --load $weights --input $file --output $outfile --num_sample_steps $num_steps
done
