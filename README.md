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
- Generate synthetic 3D TOF MRA volume, minimal code for demonstration in [local_stylegan](./local_stylegan/).
- Convert the synthetic TOF MRA to CTA, minimal code for demonstration in [local_xmodality](./local_xmodality/).
- ......... RELICT-NI ..........
- ......... Differential privacy ............. 
