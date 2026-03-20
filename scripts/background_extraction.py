import os
import sys
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

import clipasso
from clipasso import painterly_rendering
from config import model_config


def extract_background(args, input_folder, output_folder):

    """
    This function:
    1. Loads images from `input_folder` using `ImageFolder`.
    2. Resizes and center-crops each image according to `args.image_scale`.
    3. Applies the same foreground/background extraction procedure used in Clipasso
       through `clipasso.painterly_rendering.get_target`, which relies on the
       U2-Net model for saliency-based object segmentation.
    4. Saves the processed image to `output_folder` while preserving the original
       subfolder structure.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration object containing runtime arguments. It must include
        the attribute `image_scale`, which defines the resize and crop size.
    input_folder : str or Path
        Path to the root folder containing the input dataset organized in
        class subfolders, as expected by `ImageFolder`.
    output_folder : str or Path
        Path to the root folder where processed images will be saved.
    """

    transform = transforms.Compose([transforms.Resize((args.image_scale, args.image_scale)),
                                    transforms.CenterCrop(args.image_scale)])
    dataset = ImageFolder(root=input_folder, transform=transform)

    for i, (path, _) in enumerate(dataset.samples):
        input_img, _ = dataset[i]
        input_img, mask = clipasso.painterly_rendering.get_target(args, input_img)
        # Get the original file name and the structure of parent folder to save preprocessed images accordingly
        original_filename = Path(path).stem
        relative_path = Path(path).relative_to(input_folder).parent
        save_images(input_img, output_folder, original_filename, relative_path)


def save_images(img, output_folder, original_filename, relative_path):

    """
    Save a processed image to disk after normalization.
    The function accepts either a PyTorch tensor or a PIL image, converts it
    to a CPU tensor, rearranges dimensions for plotting, normalizes pixel values
    into the range [0, 1], and saves the image as a PNG file.

    Parameters
    ----------
    img : torch.Tensor or PIL.Image.Image
        The image to save. If it is a tensor, it is expected to have channel-first
        format. If it is a PIL image, it will be converted to a tensor first.
    output_folder : str or Path
        Root directory where the processed image should be stored.
    original_filename : str
        Original image filename without file extension.
    relative_path : str or Path
        Relative subdirectory path to preserve the input folder structure.
    """

    if isinstance(img, torch.Tensor):
        img = img.squeeze().to("cpu")
    else:
        img = TF.to_tensor(img).squeeze().to("cpu")
    img = img.permute(1, 2, 0)
    img_norm = (img - img.min()) / (img.max() - img.min())
    save_folder = os.path.join(output_folder, relative_path)
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{original_filename}_preprocessed.png")
    plt.imsave(save_path, img_norm.numpy())


if __name__ == "__main__":
    args = model_config.parse_arguments()
    try:
        extract_background(args, input_folder="./data/train_set", output_folder="./train_set")
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        sys.exit(1)
