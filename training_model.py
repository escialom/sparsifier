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

torch.autograd.set_detect_anomaly(True)


def train_model(args):

    # Init model, optimizer and loss
    model = PhospheneOptimizer(args=args,
                               simulator_params=dynaphos.utils.load_params("./config/config_dynaphos/params.yaml"),
                               electrode_grid=1024)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = Loss(args)

    # Prepare dataloaders
    transform = transforms.Compose([transforms.Resize((args.image_scale, args.image_scale)),
                                    transforms.CenterCrop(args.image_scale),
                                    transforms.ToTensor()])
    train_dataset = ImageFolder(root=args.train_set, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_training, shuffle=True)
    val_dataset = ImageFolder(root=args.val_set, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_validation, shuffle=False)

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
                                            epoch=0)

    # Prepare training loop
    epoch_loss_dict = {}
    training_data = {}
    val_loss_dict = {}
    validation_data = {}
    epoch_range = tqdm(range(args.num_iter))

    # Training loop
    model.train()
    for epoch in epoch_range:
        epoch_loss = 0.0
        num_batches = len(train_loader)
        for batch in train_loader:
            input_imgs,_ = batch
            input_imgs = input_imgs.to(args.device)
            optimizer.zero_grad()
            output_imgs,_ = model(input_imgs)
            losses_dict = loss_func(output_imgs, input_imgs, model.parameters(), epoch, optimizer, mode = "train")
            loss = sum(losses_dict.values())
            loss.backward()
            optimizer.step()
            # Gather batch losses to calculate epoch loss
            epoch_loss += loss
        # Save loss of each epoch
        epoch_loss = epoch_loss / num_batches
        epoch_loss_dict[epoch] = {'loss': epoch_loss}

        # Display epoch loss during training
        print(f'Epoch [{epoch+1}/{len(epoch_range)}], Loss: {epoch_loss.item():.4f}')

        # Save ongoing training data every N epochs (defined in args.save_interval)
        if epoch > 0 and epoch % args.save_interval == 0:
            # Gather current training data to the aggregated dictionary
            training_data[epoch] = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch_loss': epoch_loss_dict[epoch].get('loss')
            }

        # Validation step every N epochs (defined in args.eval_interval)
        if epoch > 0 and epoch % args.eval_interval == 0:
            val_loss = validate_model(model, val_loader, loss_func, epoch, optimizer, args.device)
            val_loss_dict[epoch] = {'loss': val_loss}
            # Store the current validation state, model, and optimizer info
            validation_data[epoch] = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss_dict[epoch].get('loss')
            }
            # Track and save the optimized (tracked) validation image
            with torch.no_grad():
                tracked_output_img, intensity = model(tracked_img)
            data_tracked_val_img = save_tracked_img(args.output_dir,
                                                    tracked_img,
                                                    tracked_output_img,
                                                    intensity,
                                                    data_tracked_val_img,
                                                    epoch)
            model.train()

    # Save data gathered during training
    torch.save(training_data, os.path.join(args.output_dir, "training_data_checkpoints.pth"))
    torch.save(validation_data, os.path.join(args.output_dir, "validation_data_checkpoints.pth"))
    torch.save(data_tracked_val_img, os.path.join(args.output_dir, "data_tracked_validation_img.pth"))

    # Save optimized images once model is trained
    val_img_after_training = {}
    model.eval()
    img_idx = 0
    for batch, _ in val_loader:
        for img_sample in batch:
            img_sample = img_sample.unsqueeze(0).to(args.device)  # Add batch dimension
            with torch.no_grad():
                output_img, intensity = model(img_sample)
            epoch_path_dir = create_dir(args.output_dir, epoch)
            filepath_img = os.path.join(epoch_path_dir, f"val_img_{img_idx}.png")
            save_images(output_img, filepath_img)
            val_img_after_training[img_idx] = {
                'input_img': img_idx,
                'output_img': output_img,
                'intensity': torch.sum(intensity).item(),
                'number_phosphenes': torch.sum(intensity > 0).item()
            }
            img_idx += 1

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
            val_losses += loss
    # Get the average validation loss of current epoch
    val_loss_epoch = val_losses / num_batches
    return val_loss_epoch


def create_dir(output_dir, epoch):
    epoch_path_dir = os.path.join(output_dir, f"epoch_{epoch}")
    if not os.path.exists(epoch_path_dir):
        os.mkdir(epoch_path_dir)
    return epoch_path_dir


def save_images(output_imgs, save_prefix='image'):
    for i in range(output_imgs.shape[0]):
        img = output_imgs[i]
        img_np = img.detach().permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        plt.imsave(f'{save_prefix}_{i}.png', img_np)


def save_tracked_img(output_dir, tracked_img_input, tracked_img_output, stimulation_intensity, dict_metadata, epoch):
    epoch_path_dir = create_dir(output_dir, epoch)
    filepath_img = os.path.join(epoch_path_dir, f"tracked_val_img_epoch_{epoch}.png")
    save_images(tracked_img_output, filepath_img)
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
