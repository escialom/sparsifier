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
        extract_background(args, input_folder="./data/val", output_folder="./data_preprocessed/val_set")
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        sys.exit(1)
