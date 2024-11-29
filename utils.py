import torch
import random
import shutil
from pathlib import Path
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from clipasso.U2Net_.model import U2NET


class MaskImgs:
    def __init__(self, args):
        self.device = args.device
        model_dir = os.path.join("./clipasso/U2Net_/saved_models/u2net.pth")
        self.net = U2NET(3, 1)
        self.net.load_state_dict(torch.load(model_dir, map_location=self.device, weights_only=True))
        self.net.to(self.device).eval()
        self.data_transforms = transforms.Compose([
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def forward(self, im_batch):

        im_batch = self.data_transforms(im_batch).to(self.device)
        with torch.no_grad():
            d1, *_ = self.net(im_batch)
        # Normalize and threshold
        pred = d1[:, 0, :, :]
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        mask = (pred >= 0.5).float()  # Binary mask
        # Resize and apply mask
        _, _, w, h = im_batch.shape
        mask = F.interpolate(mask.unsqueeze(1), size=(w, h), mode='bilinear', align_corners=False)
        mask = torch.cat([mask] * 3, dim=1)  # Expand to 3 channels for RGB compatibility
        masked_images = mask * im_batch
        masked_images[mask == 0] = 0
        masked_images = torch.clamp(masked_images, min=0.0, max=1.0)

        return masked_images, mask

    def __call__(self, im_batch):
        return self.forward(im_batch)


def mask_imgs(imgs, masks):
    masked_imgs = masks * imgs
    masked_imgs[masks == 0] = 0
    masked_imgs = torch.clamp(masked_imgs, min=0.0, max=1.0)
    return masked_imgs


def copy_random_images_per_class(source_dir, destination_dir, num_images_per_class=10):
    """
    Copies a specified number of random images per class from source directory
    (structured as source_dir/sub_dir/class_folder) to a target directory,
    preserving the class folder structure.

    Args:
        source_dir (str): Root directory containing subdirectories with class folders.
        target_dir (str): Destination directory to copy images to, preserving structure.
        num_images (int): Number of random images to copy per class.
    """
    # Create the target directory if it does not exist
    os.makedirs(destination_dir, exist_ok=True)

    # Walk through each subdirectory in source_dir looking for class folders
    for sub_dir in Path(source_dir).iterdir():
        if sub_dir.is_dir():  # Only process directories within source_dir
            for class_folder in sub_dir.iterdir():
                if class_folder.is_dir():  # Only process class folders within sub_dir
                    # List all image files in the class folder
                    image_files = [file for file in class_folder.iterdir() if file.is_file() and file.suffix.lower() in {'.jpg', '.jpeg', '.png'}]

                    # Select random images
                    selected_images = random.sample(image_files, min(num_images_per_class, len(image_files)))

                    # Create target class folder
                    target_class_folder = Path(destination_dir) / sub_dir.name / class_folder.name
                    os.makedirs(target_class_folder, exist_ok=True)

                    # Copy selected images to the target directory
                    for image in selected_images:
                        shutil.copy(image, target_class_folder / image.name)


def track_images(args, model, dataset, dataloader, input_dir, output_dir, epoch=0, at_init=False):
    """
    Generates and saves images for each input image in the specified directory,
    preserving the folder structure and adding a prefix to the filenames.
    """
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        # Access the relative paths
        for batch_idx, (input_imgs, _) in enumerate(dataloader):
            # Get output image
            input_imgs = input_imgs.to(args.device)
            output_imgs, _ = model(input_imgs)
            output_imgs = (output_imgs - output_imgs.min()) / (output_imgs.max() - output_imgs.min())
            # Save output image with correct folder structure
            for i, img in enumerate(output_imgs):
                absolute_path, _ = dataset.samples[batch_idx * len(output_imgs) + i]
                relative_path = Path(absolute_path).relative_to(input_dir)
                output_class_dir = os.path.join(output_dir, relative_path.parent)
                os.makedirs(output_class_dir, exist_ok=True)
                output_prefix = "at_init_" if at_init else f"epoch_{epoch}_"
                output_file_name = f"{output_prefix}{relative_path.stem}.png"
                output_file_path = os.path.join(output_class_dir, output_file_name)
                pil_img = Image.fromarray(img.permute(1, 2, 0).numpy())
                pil_img.save(output_file_path)


def get_memory_allocated(batch_idx, check_step, train_step='optimizer.step()'):
    if batch_idx % check_step == 0:
        print(f"Batch {batch_idx}, Memory allocated after {train_step}: {torch.cuda.memory_allocated()} bytes")


def get_layer_gradients(model, epoch, batch_idx, check_step, get_mean_grad=False):
    if batch_idx % check_step == 0:
        # Get gradients of each layer
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"Epoch {epoch}, Batch {batch_idx}, Layer {i} - Gradient Norm: {grad_norm:.6f}")
        # Get average norm gradient over all layers
        if get_mean_grad:
            grad_norms_list = [param.grad.norm().item() for param in model.parameters() if param.grad is not None]
            if grad_norms_list:
                print(f"Epoch {epoch}, Batch {batch_idx} - Average Gradient Norm: {sum(grad_norms_list) / len(grad_norms_list):.6f}")


def plot_losses(output_dir="./output", train_filename="training_data_checkpoints.pth", val_filename="validation_data_checkpoints.pth"):
    # Load the saved training and validation data and map to CPU if needed
    training_data = torch.load(os.path.join(output_dir, train_filename), map_location=torch.device('cpu'))
    validation_data = torch.load(os.path.join(output_dir, val_filename), map_location=torch.device('cpu'))
    epochs = sorted(training_data.keys())  # Assuming epochs are the keys in both dictionaries
    train_losses = [training_data[epoch]['epoch_loss'] for epoch in epochs]
    val_losses = [validation_data[epoch]['val_loss'] for epoch in epochs]
    # Plotting the training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_stim_properties(output_dir="./output", filename="data_tracked_validation_img.pth"):
    data_tracked_val_img = torch.load(os.path.join(output_dir, filename), map_location=torch.device('cpu'))
    epochs = sorted(data_tracked_val_img.keys())
    stim_intensity = [data_tracked_val_img[epoch]['stim_intensity'] for epoch in epochs]
    num_phosphenes = [data_tracked_val_img[epoch]['number_phosphenes'] for epoch in epochs]
    # Plot 1: Stim intensity per epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, stim_intensity, label='Stim Intensity', marker='o')
    plt.xticks(epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Stim Intensity')
    plt.title('Stimulation Intensity of tracked img over N epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Plot 2: Number of phosphenes per epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, num_phosphenes, label='Number of Phosphenes', marker='x', color='orange')
    plt.xticks(epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Number of Phosphenes of tracked img over N epochs')
    plt.title('Number of Phosphenes Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


def val_img_properties(output_dir="./output", img_metadata_file="val_img_metadata.pth"):
    img_metadata_path = os.path.join(output_dir, img_metadata_file)
    val_img_after_training = torch.load(img_metadata_path, map_location=torch.device('cpu'))
    img_indices = sorted(val_img_after_training.keys())
    intensities = [val_img_after_training[img_idx]['intensity'] for img_idx in img_indices]
    num_phosphenes = [val_img_after_training[img_idx]['number_phosphenes'] for img_idx in img_indices]
    # Plot 1: Intensity per image index
    plt.figure(figsize=(10, 6))
    plt.bar(img_indices, intensities, color='blue', alpha=0.7)
    plt.xlabel('Image Index')
    plt.ylabel('Intensity')
    plt.title('Stimulation Intensity per Image')
    plt.xticks(img_indices)
    plt.grid(axis='y')
    plt.show()
    # Plot 2: Number of Phosphenes per image index
    plt.figure(figsize=(10, 6))
    plt.bar(img_indices, num_phosphenes, color='orange', alpha=0.7)
    plt.xlabel('Image Index')
    plt.ylabel('Number of Phosphenes')
    plt.title('Number of Phosphenes per Image')
    plt.xticks(img_indices)
    plt.grid(axis='y')
    plt.show()


def plot_optimized_img(output_imgs):
    out = output_imgs.squeeze(0)
    out = out.permute(1, 2, 0)
    numpy_img = out.numpy()
    numpy_img = (numpy_img - numpy_img.min()) / (numpy_img.max() - numpy_img.min())
    plt.imshow(numpy_img, cmap='gray')
    plt.axis('off')  # Turn off axis labels
    plt.show()


def save_images(output_imgs, save_prefix, input_filename, epoch_idx, at_init=False):
    for i in range(output_imgs.shape[0]):
        img = output_imgs[i]
        img_np = img.detach().permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        if at_init:
            filename_output = f"{input_filename}_phos_model_init.png"
        else:
            filename_output = f"{input_filename}_phos_{epoch_idx}_{i}.png"
        plt.imsave(os.path.join(save_prefix, filename_output), img_np)


def create_dir(output_dir, epoch):
    epoch_path_dir = os.path.join(output_dir, f"epoch_{epoch}")
    if not os.path.exists(epoch_path_dir):
        os.mkdir(epoch_path_dir)
    return epoch_path_dir


def save_tracked_img(output_dir, tracked_img_input, tracked_img_output, stimulation_intensity, dict_metadata, val_dataset, tracked_img_idx, input_folder, epoch, at_init=False):
    if isinstance(val_dataset, torch.utils.data.DataLoader):
        val_dataset = val_dataset.dataset
    epoch_path_dir = create_dir(output_dir, epoch)
    img_path, _ = val_dataset.samples[tracked_img_idx]
    relative_path = os.path.relpath(img_path, input_folder)
    input_filename = os.path.splitext(os.path.basename(relative_path))[0]
    output_img_dir = os.path.join(epoch_path_dir, os.path.dirname(relative_path))
    os.makedirs(output_img_dir, exist_ok=True)
    save_images(tracked_img_output, output_img_dir, input_filename, epoch, at_init)
    dict_metadata[epoch] = {
        'input_img': tracked_img_input,
        'output_img': tracked_img_output,
        'stim_intensity': torch.sum(stimulation_intensity).item(),
        'number_phosphenes': torch.sum(stimulation_intensity > 0).item()
    }
    return dict_metadata


def track_val_imgs(args, n_img_tracked, dataloader, model, data_tracked_val_imgs, device='cpu', seed=None):
    """
    Track a specified number of validation images during training.

    Args:
        n_img_tracked (int): Number of validation images to track.
        dataloader (DataLoader): DataLoader for the validation dataset.
        model (torch.nn.Module): The neural network model.
        device (str): Device to perform computations on ('cpu' or 'cuda').
        seed (int, optional): Seed for reproducibility of random image selection.

    Returns:
        dict: Dictionary containing tracked image data.
    """
    # Set random seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)

    tracked_img_indices = random.sample(range(len(dataloader.dataset)), n_img_tracked)
    tracked_imgs = []

    # Retrieve and prepare each tracked image
    for idx in tracked_img_indices:
        tracked_img, _ = dataloader.dataset[idx]
        tracked_img = tracked_img.unsqueeze(0).to(device)  # Add batch dimension
        tracked_imgs.append((idx, tracked_img))
        if idx not in data_tracked_val_imgs:
            data_tracked_val_imgs[idx] = {}

    # Set model to evaluation mode and track images
    model.eval()
    with torch.no_grad():
        for idx, tracked_img in tracked_imgs:
            tracked_output_img, intensity = model(tracked_img)
            data_tracked_val_imgs[idx] = save_tracked_img(args.output_dir,
                                                          tracked_img,
                                                          tracked_output_img,
                                                          intensity,
                                                          data_tracked_val_imgs[idx],
                                                          dataloader,
                                                          idx,
                                                          input_folder=args.val_set,
                                                          epoch=0,
                                                          at_init=True)

    return data_tracked_val_imgs, tracked_imgs


def normalized_rescaling(phosphene_placement_map, max_stimulation_intensity=1):
    """Normalize <img> and rescale the pixel intensities in the range [0, <stimulus_scale>].
    <stimulus_scale> is defined in the parameter file.
    The output image represents the stimulation intensity map.
    return: image with rescaled pixel values (stimulation intensity map in Ampères)."""

    img_norm = (phosphene_placement_map - phosphene_placement_map.min()) / (phosphene_placement_map.max() - phosphene_placement_map.min())
    return img_norm * max_stimulation_intensity


def make_lr_lambda(warm_up_epochs):
    def lr_lambda(epoch):
        if epoch < warm_up_epochs:
            return float(epoch + 1) / float(warm_up_epochs)
        return 1.0
    return lr_lambda