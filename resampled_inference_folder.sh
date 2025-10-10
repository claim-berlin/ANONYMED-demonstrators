#!/bin/bash
weights=dit.safetensors
num_steps=1
mkdir -p output

for file in "$1"/*
do
    outfile="output/${file##*/}"
    echo "$file -> $outfile"

    python3 resampled_inference.py --input $file --output $outfile --load $weights --arch dit --bfloat16
done
