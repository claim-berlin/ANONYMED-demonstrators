# Projects

* `diff_private` - differential privacy 2d generative model example on 32x32 atlas slices
* `xmodality` - tof-mra to cta diffusion translation
* `stylegan` - 3d stylegan network


# Synthetic TOF MRA generation
Generates a 3D 256x256x4 cerebral TOF-MRA volume centered on the Circle of Willis. The voxel spacing is 0.62x062x0.62 millimeters. Local root for demonstrator scripts: [local_stylegan](./local_stylegan/).

## Model weights
- Download model weights from: .......................
- Place model weights here: ./local_stylegan/models/256_256_4/model_17.pt

## Environment
- Create environment for StyleGAN: 
```bash  
conda env create -f environment_styleGAN.yml   
conda activate StyleGAN_CoW_3D_anonymed   
```

## Usage

```bash
    ./stylegan_inference_tofmra.sh [TRUNCATION_PSI] [NUM_VOLUMES] [OUTPUT_DIR]
```

| Positional arg   | Description                               | Default            |
|------------------|-------------------------------------------|--------------------|
| `TRUNCATION_PSI` | StyleGAN truncation parameter (ψ)          | `1`                |
| `NUM_VOLUMES`    | Number of volumes to generate              | `1`                |
| `OUTPUT_DIR`     | Output directory for NIfTI files           | `output_TOF` |


### Default parameters
Sets ψ = 1, generates a single volume, saves to `output_TOF`:

```bash
    ./stylegan_inference_tofmra.sh
```

### User-specified parameters
Order: 1) truncation ψ, 2) number of volumes, 3) output directory:

```bash
    ./stylegan_inference_tofmra.sh 1 3 ./output_TOF
```

## Output

Files are written to `output_TOF` with names like:

    TOF_MRA_000.nii.gz
    TOF_MRA_001.nii.gz


This repository implements the methodology described in:

Aydin, Orhun Utku, Adam Hilbert, Alexander Koch, Felix Lohrke, Jana Rieger, Satoru Tanioka, and Dietmar Frey. “Generative Modeling of the Circle of Willis Using 3D-StyleGAN.” NeuroImage, November 23, 2024, 120936. https://doi.org/10.1016/j.neuroimage.2024.120936.

Link to full repository can be found at [stylegan](./stylegan/).

# xmodality

Takes an unprocessed --- real or synthetic --- TOF-MRA image (normal MR range) and returns a CT in [-50,350] windowed range. Local root for demonstrator scripts: [local_xmodality](./local_xmodality/).


## Model weights

- Download model weights from huggingface: wget https://huggingface.co/alexander-koch/xmodality/resolve/main/dit.safetensors
- Place model weights here: ./local_xmodality/dit.safetensors

## Environment
- Create environment for cross modality from TOF-MRA to CTA:
```bash
conda env create -f environment_xmodality.yml   
conda activate xmodality_anonymed
```

## Usage
Run multiple at once:
Runs inference on the entire folder `output_TOF` and outputs the images into `output_CTA`.
Tweak the script for custom weight paths, architecture, etc.

macOS

```bash
chmod u+x mlx_inference_folder.sh
./mlx_inference_folder.sh ./output_TOF
```

Linux

```bash
chmod u+x resampled_inference_folder.sh
./resampled_inference_folder.sh ./output_TOF
```

## Output

Files are written to `output_CTA` with names like:

    CTA_000.nii.gz
    CTA_001.nii.gz


This repository implements the methodology described in:

Alexander Koch, Aydin, Orhun Utku, Adam Hilbert, Jana Rieger, Satoru Tanioka, Fujimaro Ishida and Dietmar Frey. “Cross-modality image synthesis from TOF-MRA to CTA using diffusion-based models.” Medical Image Analysis, October, 2025, 103722. https://doi.org/10.1016/j.media.2025.103722.

Link to full repository can be found at [xmodality](./xmodality/).

### Example synthetic TOF (left) and converted synthetic CTA (right)

![Example_TOF_and_CTA](./images/TOF_CTA_example.jpg)