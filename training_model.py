import os
import random
import sys
import torch
import traceback

import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import model_config
import dynaphos
from model import PhospheneOptimizer
from clipasso.models.loss import Loss


def train_model(args):

    # Prepare dataloaders
    train_dataset = ImageFolder(root=args.train_set, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_training, shuffle=True)
    val_dataset = ImageFolder(root=args.val_set, transform=transforms.ToTensor())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_validation, shuffle=False)

    # Init model, optimizer and loss
    model = PhospheneOptimizer(args=args,
                               simulator_params=dynaphos.utils.load_params("./config/config_dynaphos/params.yaml"),
                               electrode_grid=1024,
                               batch_size=args.batch_size_training)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = Loss(args)

    # Keep track of a random validation image
    random.seed(args.seed)
    tracked_img_idx = random.randint(0, len(val_loader))
    tracked_img, _ = val_loader.dataset[tracked_img_idx]
    tracked_img = tracked_img.unsqueeze(0).to(args.device) # Add batch dimension
    data_tracked_val_img = {}
    # Get optimized tracked image at epoch 0
    model.eval()
    with torch.no_grad():
        tracked_output_img, intensity = model(tracked_img)
    data_tracked_val_img = save_tracked_img(args.output_dir,
                                            tracked_img,
                                            tracked_output_img,
                                            intensity,
                                            data_tracked_val_img,
                                            val_dataset,
                                            tracked_img_idx,
                                            input_folder=args.val_set,
                                            epoch=0)

    # Prepare training loop
    epoch_loss_dict = {}
    training_data = {}
    val_loss_dict = {}
    validation_data = {}
    prev_val_loss_checked = None
    min_val_loss_diff = 1e-5
    epoch_range = tqdm(range(args.num_iter))

    # Training loop
    model.train()
    for epoch in epoch_range:
        epoch_loss = 0.0
        num_batches = len(train_loader)
        batch_idx = 0
        for batch in train_loader:
            input_imgs, _ = batch
            input_imgs = input_imgs.to(args.device)
            optimizer.zero_grad()
            output_imgs, _ = model(input_imgs)
            losses_dict = loss_func(output_imgs, input_imgs, model.parameters(), epoch, optimizer, mode = "train")
            loss = sum(losses_dict.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            batch_idx += 1
        # Save loss of each epoch
        epoch_loss = epoch_loss / num_batches
        epoch_loss_dict[epoch] = {'loss': epoch_loss}

        # Save ongoing training data every epoch
        training_data[epoch] = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_loss': epoch_loss_dict[epoch].get('loss')
        }
        print(f'Epoch [{epoch}/{len(epoch_range)}], Training Loss: {epoch_loss:.5f}')
        # Save ongoing validation data every epoch
        val_loss = validate_model(model, val_loader, loss_func, epoch, optimizer, args.device)
        val_loss_dict[epoch] = {'loss': val_loss}
        # Store the current validation state, model, and optimizer info
        validation_data[epoch] = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss_dict[epoch].get('loss')
        }
        print(f'Epoch [{epoch}/{len(epoch_range)}], Validation Loss: {val_loss:.5f}')

        # Every N epochs (defined in args.check_interval), save tracked validation image and check for early stopping
        if epoch > 0 and epoch % args.check_interval == 0:
            # Save the optimized (tracked) validation image and register its metadata
            with torch.no_grad():
                tracked_output_img, intensity = model(tracked_img)
            data_tracked_val_img = save_tracked_img(args.output_dir,
                                                    tracked_img,
                                                    tracked_output_img,
                                                    intensity,
                                                    data_tracked_val_img,
                                                    val_dataset,
                                                    tracked_img_idx,
                                                    input_folder=args.val_set,
                                                    epoch=epoch)
            # Stop the training loop if the difference is less than the threshold
            if prev_val_loss_checked is not None:
                val_loss_diff = abs(val_loss - prev_val_loss_checked)
                if val_loss_diff < min_val_loss_diff:
                    # Update files before breaking the training
                    torch.save(training_data, os.path.join(args.output_dir, "training_data_checkpoints.pth"))
                    torch.save(validation_data, os.path.join(args.output_dir, "validation_data_checkpoints.pth"))
                    torch.save(data_tracked_val_img, os.path.join(args.output_dir, "data_tracked_validation_img.pth"))
                    print(f"Stopping training early at epoch {epoch} due to minimal validation loss improvement.")
                    break
            prev_val_loss_checked = val_loss

        # Update files
        torch.save(training_data, os.path.join(args.output_dir, "training_data_checkpoints.pth"))
        torch.save(validation_data, os.path.join(args.output_dir, "validation_data_checkpoints.pth"))
        torch.save(data_tracked_val_img, os.path.join(args.output_dir, "data_tracked_validation_img.pth"))

    # Get optimized validation images after training
    val_img_after_training = {}
    model.eval()
    n_val_samples = 0
    for batch_idx, (batch, _) in enumerate(val_loader):
        for img_idx, img_sample in enumerate(batch):
            img_sample = img_sample.unsqueeze(0).to(args.device)
            with torch.no_grad():
                output_img, intensity = model(img_sample)
            relative_img_path, _ = val_loader.dataset.samples[batch_idx * len(batch) + img_idx]
            input_filename = os.path.splitext(os.path.basename(relative_img_path))[0]
            val_img_post_training_dir = os.path.join(args.output_dir, "val_img_post_training")
            output_img_dir = os.path.join(val_img_post_training_dir, os.path.dirname(relative_img_path))
            os.makedirs(output_img_dir, exist_ok=True)
            save_images(output_img, output_img_dir, input_filename, epoch)
            val_img_after_training[n_val_samples] = {
                'input_img': img_idx,
                'output_img': output_img,
                'intensity': torch.sum(intensity).item(),
                'number_phosphenes': torch.sum(intensity > 0).item()
            }
            n_val_samples += 1

    # Save metadata of validation images
    torch.save(val_img_after_training, os.path.join(args.output_dir, "data_val_imgs_after_training.pth"))

    return model.state_dict()


def validate_model(model, validation_loader, loss_func, epoch, optimizer, device):
    model.eval()
    val_losses = 0.0
    num_batches = len(validation_loader)
    for batch in validation_loader:
        input_imgs, _ = batch
        input_imgs = input_imgs.to(device)
        with torch.no_grad():
            output_imgs, _ = model(input_imgs)
            losses_dict = loss_func(output_imgs, input_imgs, model.parameters(), epoch, optimizer, mode="eval")
            loss = sum(losses_dict.values())
            val_losses += loss.item()
    # Get the average validation loss of current epoch
    val_loss_epoch = val_losses / num_batches
    # Set model back to training mode
    model.train()
    return val_loss_epoch


def create_dir(output_dir, epoch):
    epoch_path_dir = os.path.join(output_dir, f"epoch_{epoch}")
    if not os.path.exists(epoch_path_dir):
        os.mkdir(epoch_path_dir)
    return epoch_path_dir


def save_images(output_imgs, save_prefix, input_filename, epoch_idx):
    for i in range(output_imgs.shape[0]):
        img = output_imgs[i]
        img_np = img.detach().permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        filename_output = f"{input_filename}_phos_{epoch_idx}_{i}.png"
        plt.imsave(os.path.join(save_prefix, filename_output), img_np)


def save_tracked_img(output_dir, tracked_img_input, tracked_img_output, stimulation_intensity, dict_metadata, val_dataset, tracked_img_idx, input_folder, epoch):
    epoch_path_dir = create_dir(output_dir, epoch)
    img_path, _ = val_dataset.samples[tracked_img_idx]
    relative_path = os.path.relpath(img_path, input_folder)
    input_filename = os.path.splitext(os.path.basename(relative_path))[0]
    output_img_dir = os.path.join(epoch_path_dir, os.path.dirname(relative_path))
    os.makedirs(output_img_dir, exist_ok=True)
    save_images(tracked_img_output, output_img_dir, input_filename, epoch)
    dict_metadata[epoch] = {
        'input_img': tracked_img_input,
        'output_img': tracked_img_output,
        'stim_intensity': torch.sum(stimulation_intensity).item(),
        'number_phosphenes': torch.sum(stimulation_intensity > 0).item()
    }
    return dict_metadata


if __name__ == "__main__":
    args = model_config.parse_arguments()
    final_config = vars(args)
    try:
        model = train_model(args)
        torch.save(model, f"{args.output_dir}/model.pth")
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
    np.save(f"{args.output_dir}/model_config.npy", final_config)
