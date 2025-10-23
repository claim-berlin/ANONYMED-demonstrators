#!/bin/bash
# Defaults fore truncation psi and num_volumes are 1. 
# Usage: ./run_generate_tof_mra.sh [TRUNCATION_PSI] [NUM_VOLUMES] [OUTPUT_DIR]
PY="./TOF_MRA_generation/generate_TOF_MRA_with_3D_styleGAN.py"

TRUNCATION_PSI="${1:-1}"
NUM_VOLUMES="${2:-1}"
OUTPUT_DIR="${3:-./output_tof_mra}"

python3 "$PY" \
  --truncation-psi "$TRUNCATION_PSI" \
  --num-volumes "$NUM_VOLUMES" \
  --output-dir "$OUTPUT_DIR"

