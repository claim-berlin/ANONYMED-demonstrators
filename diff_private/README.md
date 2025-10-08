# Diff private example

* Transformer architecture not converging
* U-Net very slow and OOM for batch sizes >512
* MLP Mixer converges somewhat, speed fast (about 20h)
* large batch sizes (≥ 2048), constant learning rate, 1e−4, 32 × 32 image resolution
* so far: DP-SGD, DP-ADAM, L2 norm grad clip 1, noise multiplier 1.3, δ= 1e−6, account ε until target is reached

# Run

```bash
python3 train_unconditional_dp_atlas.py --sample --bfloat16 --arch mlp --load ../weights/atlas_dp.pkl
```

# License

Copyright (c) Alexander Koch 2025
