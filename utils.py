import torch
import os
import matplotlib.pyplot as plt


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
    plt.xticks(epochs)
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