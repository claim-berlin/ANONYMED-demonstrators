import os
import numpy as np
import torch
import nibabel as nib
from stylegan2_pytorch import ModelLoader
import argparse

"""
Orhun Utku Aydin, Adam Hilbert, Alexander Koch, Felix Lohrke, Jana Rieger, Satoru Tanioka, Dietmar Frey,
Generative modeling of the Circle of Willis using 3D-StyleGAN,
NeuroImage,
Volume 304,
2024,
120936,
ISSN 1053-8119,
https://doi.org/10.1016/j.neuroimage.2024.120936.
(https://www.sciencedirect.com/science/article/pii/S1053811924004336)
"""

# CONFIG ---
# Model weights file should be placed inside {top_dir)/models/{project_name}/model_{epoch}.pt :
# Example model_path = "./models/256_256_4/model_17.pt"
top_dir = "./TOF_MRA_generation"
project_name = "256_256_4"
epoch = 17

# Training-time architecture settings, matches the provided checkpoint
fmap_max = 512
network_capacity = 8

# Number of synthetic TOF MRAs to generate of size 256x256x4
num_volumes = 1

# Truncation parameter (1.0 = untruncated)
truncation_psi = 1.0

# Template NIfTI for affine and voxel spacing
template_path = "./TOF_MRA_generation/sample_data/TOF_template_NITRC_BET.nii.gz"

# Output directory and filename
filename_prefix = "TOF_MRA"

# Optional seed for reproducibility
base_seed = None


# GENERATION FUNCTION
def generate_tof_mra_samples(
    top_dir: str,
    project_name: str,
    epoch: int,
    fmap_max: int,
    network_capacity: int,
    num_images: int,
    truncation_psi: float,
    template_path: str,
    output_dir: str,
    filename_prefix: str,
    base_seed: int | None = None,
) -> list[str]:
    """
    Generate 3D TOF-MRA volumes using the 3D StyleGAN2 model and save as NIfTI.

    Returns:
        List of absolute file paths to the saved .nii.gz files.
    """
    # setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    if base_seed is not None:
        torch.manual_seed(base_seed)
        np.random.seed(base_seed)

    affine = nib.load(template_path).affine

    # load model uses EMA generator from ModelLoader
    loader = ModelLoader(
        base_dir=top_dir,
        name=project_name,
        fmap_max=fmap_max,
        network_capacity=network_capacity,
        load_from=epoch
    )

    S = loader.model.image_size
    saved_paths: list[str] = []

    for i in range(num_images):
        # latent -> styles
        z = torch.randn(1, 512, device=device)
        styles = loader.noise_to_styles(z, trunc_psi=truncation_psi)

        # external noise volume for generator
        ext_noise = torch.rand(1, S, S, S, 1, device=device)

        # generate and clamp to [0, 1]
        images = loader.styles_to_images_ext_noise(styles, ext_noise).clamp_(0.0, 1.0)

        # convert to numpy and save as NIfTI
        img_np = images.squeeze(0).permute(2, 3, 1, 0).squeeze(-1).detach().cpu().numpy()
        img_np = np.clip(img_np, 0, 1)


        out_path = os.path.abspath(os.path.join(output_dir, f"{filename_prefix}_{i:03d}.nii.gz"))
        nib.save(nib.Nifti1Image(img_np, affine), out_path)
        saved_paths.append(out_path)

    return saved_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic 3D TOF-MRA volumes with 3D-StyleGAN2 and save as NIfTI."
    )
    parser.add_argument(
        "--truncation-psi",
        type=float,
        required=True,
        dest="truncation_psi",
        help="Truncation parameter for StyleGAN (e.g., 1.0 for untruncated, 0.7 for stronger truncation).",
    )
    parser.add_argument(
        "--num-volumes",
        type=int,
        required=True,
        dest="num_volumes",
        help="Number of 3D TOF MRA volumes to generate (must be >= 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        dest="output_dir",
        help="Directory to save generated NIfTI files.",
    )
    args = parser.parse_args()

    # simple validation
    if args.num_volumes < 1:
        raise ValueError("--num-volumes must be >= 1")
    if not (0.0 <= args.truncation_psi <= 2.0):
        raise ValueError("--truncation-psi must be between 0.0 and 2.0, recommended: 0.5 to 1")

    return args


if __name__ == "__main__":
    args = parse_args()

    saved_files = generate_tof_mra_samples(
        top_dir=top_dir,
        project_name=project_name,
        epoch=epoch,
        fmap_max=fmap_max,
        network_capacity=network_capacity,
        num_images=args.num_volumes,
        truncation_psi=args.truncation_psi,
        template_path=template_path,
        output_dir=args.output_dir,
        filename_prefix=filename_prefix,
        base_seed=base_seed,
    )

    print("Saved files:")
    for p in saved_files:
        print(" -", p)
