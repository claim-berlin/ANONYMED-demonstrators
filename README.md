# Projects

* `diff_private` - differential privacy 2d generative model example on 32x32 atlas slices
* `xmodality` - tof-mra to cta diffusion translation
* `stylegan` - 3d stylegan network

Each subfolder contains a README detailing the installation and usage of the tools.

# Install

```bash
git clone https://github.com/claim-berlin/ANONYMED-demonstrators.git
cd ANONYMED-demonstrators
git submodule update --init --recursive
```

# DEMONSTRATOR SUMMARY 
After configuring the environments and downloading model weights:
- Generate synthetic 3D TOF MRA volume   
./stylegan_inference_tofmra.sh
- Convert the synthetic TOF MRA to CTA  
./resampled_inference_folder.sh /home/orhun/PycharmProjects/ANONYMED/ANONYMED-demonstrators/tof_mra_output
- ......... RELICT-NI ..........
- ......... Differential privacy ............. 

# Synthetic TOF MRA generation
Generates a 3D 256x256x4 cerebral TOF-MRA volume centered on the Circle of Willis. The voxel spacing is 0.62x062x0.62 millimeters.

## Model weights
- Download model weights from: .......................
- Place model weights here: ./TOF_MRA_generation/models/256_256_4/model_17.pt

## Environment
- Create environment for StyleGAN:   
conda env create -f environment_styleGAN.yml   
conda activate StyleGAN_CoW_3D_anonymed   
  

## Usage

    ./stylegan_inference_tofmra.sh [TRUNCATION_PSI] [NUM_VOLUMES] [OUTPUT_DIR]

| Positional arg   | Description                               | Default            |
|------------------|-------------------------------------------|--------------------|
| `TRUNCATION_PSI` | StyleGAN truncation parameter (ψ)          | `1`                |
| `NUM_VOLUMES`    | Number of volumes to generate              | `1`                |
| `OUTPUT_DIR`     | Output directory for NIfTI files           | `./tof_mra_output` |


### Default parameters
Sets ψ = 1, generates a single volume, saves to `./tof_mra_output`:

    ./stylegan_inference_tofmra.sh

### User-specified parameters
Order: 1) truncation ψ, 2) number of volumes, 3) output directory:

    ./stylegan_inference_tofmra.sh 1 3 ./tof_mra_output

## Output

Files are written to `OUTPUT_DIR` with names like:

    TOF_MRA_000.nii.gz
    TOF_MRA_001.nii.gz


This repository implements the methodology described in:

Aydin, Orhun Utku, Adam Hilbert, Alexander Koch, Felix Lohrke, Jana Rieger, Satoru Tanioka, and Dietmar Frey. “Generative Modeling of the Circle of Willis Using 3D-StyleGAN.” NeuroImage, November 23, 2024, 120936. https://doi.org/10.1016/j.neuroimage.2024.120936.

# xmodality

Takes an unprocessed --- real or synthetic --- TOF-MRA image (normal MR range) and returns a CT in [-50,350] windowed range.


## Model weights

- Download model weights from huggingface: wget https://huggingface.co/alexander-koch/xmodality/resolve/main/dit.safetensors
- Place model weights here: ./dit.safetensors

## Environment
- Create environment for cross modality from TOF-MRA to CTA:  
conda env create -f environment_xmodality.yml   
conda activate xmodality_anonymed   

## Usage
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

