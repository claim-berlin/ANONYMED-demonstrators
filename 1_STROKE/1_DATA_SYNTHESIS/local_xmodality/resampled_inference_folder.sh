#!/bin/bash
weights=dit.safetensors
num_steps=1
mkdir -p output_CTA

for file in "$1"/*
do
    outfile="output_CTA/${file##*/}"
    echo "$file -> $outfile"
    python3 xmodality/resampled_inference.py --input $file --output $outfile --load $weights --arch dit --bfloat16
done
