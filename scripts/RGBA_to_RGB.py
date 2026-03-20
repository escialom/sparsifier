import torch
import os
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from torchvision import transforms


def RGBA_to_RGB(input_path, output_path):
    """
    Convert PNG images from RGBA to RGB format.

    This function recursively scans all `.png` files in the input directory,
    detects images with an alpha channel (RGBA), and converts them to RGB.
    Transparent pixels (alpha = 0) are replaced with white (255) in all RGB channels.
    The processed images are saved to the output directory while preserving
    the original subfolder structure.

    Parameters
    ----------
    input_path : str or Path
        Path to the input directory containing PNG images. The function
        searches recursively through all subdirectories.

    output_path : str or Path
        Path to the output directory where converted images will be saved.
        Subfolder structure from the input directory is preserved.
    """

    input_folder = Path(input_path)
    output_folder = Path(output_path)
    for file in input_folder.rglob("*.png"):
        subfolder = file.parent.name
        image = Image.open(file)
        transform = transforms.PILToTensor()
        image = transform(image)
        # if the image has 4 channels, reduce it to 3
        if image.size(0) == 4:
            for i in range(0, 3):
                image[i, :, :] = torch.where(image[3, :, :] == 0,
                                             torch.tensor(255, dtype=torch.uint8),
                                             image[i, :, :],)
        image = image[0:3, :, :]
        image = image.permute(1,2,0)
        image = image.numpy()
        os.makedirs(output_folder / subfolder, exist_ok=True)
        filepath = output_folder / subfolder / file.name
        plt.imsave(filepath, image)


if __name__ == "__main__":
    input_path = "./input"
    output_path = "./output/RGB"
    RGBA_to_RGB(input_path, output_path)