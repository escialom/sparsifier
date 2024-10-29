import torch
import random
import os
import matplotlib.pyplot as plt

import clipasso.sketch_utils as clipasso_utils


def mask_input_imgs(args, input_imgs):
    masked_img_batch = torch.empty((input_imgs.shape[0], 3, args.image_scale, args.image_scale), device=args.device)
    for i in range(input_imgs.shape[0]):
        masked_im, mask = clipasso_utils.get_mask_u2net(args, input_imgs[i])
        masked_im = masked_im.permute(2, 0, 1).unsqueeze(0)
        masked_img_batch[i] = masked_im

    return masked_img_batch


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