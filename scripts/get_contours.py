from src.ContourExtract import ContourExtract
from torchvision import transforms
from config import model_config
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import os
import shutil


args = model_config.parse_arguments()
gray_scale = transforms.Grayscale(num_output_channels=1)
padding = transforms.Pad(padding=args.padding_pix*args.sigma_kernel, fill=args.padding_color)
cropping = transforms.CenterCrop(args.image_scale)
get_contours = ContourExtract(n_orientations=args.n_orientations, sigma_kernel=args.sigma_kernel, lambda_kernel=args.lambda_kernel)
contours_batch = torch.empty(1, 1, args.image_scale, args.image_scale)
transform = transforms.Compose([transforms.Resize((args.image_scale, args.image_scale)),
                                                   transforms.ToTensor()])

def save_stimuli(dataset, output_imgs, relative_path_imgs, output_path, output_filename):
    subfolder = relative_path_imgs.parent
    # Build the output path with same subfolder structure
    output_img_dir = Path(output_path) / subfolder.relative_to(dataset.root)
    output_img_dir.mkdir(parents=True, exist_ok=True)
    if output_imgs.ndim == 3 and output_imgs.shape[0] == 3:
        output_imgs = output_imgs.permute(1, 2, 0)
    output_imgs = (output_imgs - output_imgs.min()) / (output_imgs.max() - output_imgs.min())
    plt.imsave(f"{output_img_dir}/{output_filename}", output_imgs.numpy(), cmap='gray')
    return output_img_dir


def contours_extraction(img_batch, contours_batch, requires_grad=False):
    with torch.no_grad():
        for i in range(img_batch.shape[0]):
            img = gray_scale(img_batch[i])
            img = padding(img)
            contours = get_contours(img.unsqueeze(0))
            # Collapse the contours extracted in each orientation
            contours = get_contours.create_collapse(contours)
            # Remove padding to get back to dimensions of input img
            contours = cropping(contours)
            contours_batch[i] = contours
        # Normalize each image of the batch between 0 and 1
        contours_batch = ((contours_batch - contours_batch.amin(dim=(1, 2, 3), keepdim=True)) / (contours_batch.amax(dim=(1, 2, 3), keepdim=True) - contours_batch.amin(dim=(1, 2, 3), keepdim=True) + 1e-8))
    contours_batch.requires_grad = requires_grad
    return contours_batch.to(args.device)

def copy_and_rename_images(input_folder: str, output_folder: str):
    """
    Recursively copies all images from `input_folder` (including subfolders)
    to `output_folder`, renaming them as '{original_filename}_RGB.png'.
    If multiple files have the same name, adds a numeric suffix.
    """
    os.makedirs(output_folder, exist_ok=True)
    supported_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith(supported_exts):
                src_path = os.path.join(root, filename)
                base_name = os.path.splitext(filename)[0]
                dst_filename = f"{base_name}_RGB.png"
                dst_path = os.path.join(output_folder, dst_filename)

                # Ensure unique name if file already exists
                counter = 1
                while os.path.exists(dst_path):
                    dst_filename = f"{base_name}_RGB_{counter}.png"
                    dst_path = os.path.join(output_folder, dst_filename)
                    counter += 1

                shutil.copy2(src_path, dst_path)

copy_and_rename_images(
    input_folder=args.target_path,
    output_folder=args.output_path
)
dataset = ImageFolder(root=args.target_path, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
n_iter = 0
for batch in loader:
    input_imgs, _ = batch
    input_imgs = input_imgs.to(args.device)
    contours_batch = contours_extraction(input_imgs, contours_batch, requires_grad=False)
    output_imgs = contours_batch.squeeze().squeeze()
    relative_path, _ = loader.dataset.samples[n_iter]
    relative_path = Path(relative_path)
    input_filename = relative_path.name
    filename = f"{relative_path.stem}_contours.png"
    output_img_dir = save_stimuli(dataset, output_imgs, relative_path, args.output_path, filename)
    n_iter += 1

#dataset = ImageFolder(root=args.target_path, transform=transform)
#loader = DataLoader(dataset, batch_size=1, shuffle=False)
#n_iter = 0
#for batch in loader:
#    input_imgs, _ = batch
#    input_imgs = input_imgs.to(args.device)
#    for i in range(input_imgs.shape[0]):
#        gray_img = gray_scale(input_imgs[i])
#    output_imgs = gray_img.squeeze().squeeze()
#    output_imgs = (output_imgs - output_imgs.min()) / (output_imgs.max() - output_imgs.min())
#    relative_path, _ = loader.dataset.samples[n_iter]
#    relative_path = Path(relative_path)
#    input_filename = relative_path.name
#    filename = f"{relative_path.stem}_grayscaled.png"
#    output_img_dir = save_stimuli(dataset, output_imgs, relative_path, args.output_path, filename)
#    n_iter += 1

