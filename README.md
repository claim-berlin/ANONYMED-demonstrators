# Projects

* `diff_private` - differential privacy 2d generative model example on 32x32 atlas slices
* `xmodality` - tof-mra to cta diffusion translation
* `stylegan` - 3d stylegan network

Each subfolder contains a README detailing the installation and usage of the tools.

# xmodality

Takes an unprocessed TOF-MRA image (normal MR range) and returns a CT in [-50,350] windowed range.

Run multiple at once:
Runs inference on the entire folder `source` and outputs the images into `output`.
Tweak the script for custom weight paths, architecture, etc.

macOS

```bash
chmod u+x mlx_inference_folder.sh
./mlx_inference_folder.sh source
```

Linux

```bash
chmod u+x resampled_inference_folder.sh
./resampled_inference_folder.sh source
```

