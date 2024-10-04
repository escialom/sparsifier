import os
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
import clipasso
import dynaphos
from model import PhospheneOptimizer
from clipasso.models.loss import Loss
from clipasso import painterly_rendering

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
    val_dataset = ImageFolder(root=args.val_set, transform=transforms.ToTensor())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_validation, shuffle=False)

    # Keep track of an image during training
    tracked_img_idx = 1
    tracked_img, _ = val_loader.dataset[tracked_img_idx]
    tracked_img = tracked_img.unsqueeze(0).to(args.device) # Add batch dimension

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
        print(f'Epoch [{epoch}/{len(epoch_range)}], Loss: {epoch_loss.item():.4f}')

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
            val_loss, num_phosphenes = validate_model(model, val_loader, loss_func, epoch, optimizer, args.device)
            val_loss_dict[epoch] = {'loss': val_loss}
            # Store the current validation state, model, and optimizer info
            validation_data[epoch] = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss_dict[epoch].get('loss')
            }
            # Track and save the optimized (tracked) validation image
            with torch.no_grad():
                tracked_output_img,_ = model(tracked_img)
            epoch_path_dir = create_dir(args.output_dir, epoch)
            filepath_img = os.path.join(epoch_path_dir, f"tracked_val_img_epoch_{epoch}.png")
            save_images(tracked_output_img, filepath_img)
            model.train()

    # Save the data gathered during training
    filepath_data = os.path.join(args.output_dir, "training_data_checkpoints.pth")
    torch.save(training_data, filepath_data)
    # Get training data of the last epoch
    training_data_last_epoch = {
        'epoch_loss': epoch_loss_dict,
        'val_loss': validation_data.get('val_loss'),
        'final_epoch': args.num_iter-1,
        'args': args
    }
    # Get the optimized images last epoch training
    save_images(output_imgs, save_prefix=os.path.join(args.output_dir,"output_images_training"))

    return model.state_dict(), training_data_last_epoch


def validate_model(model, validation_loader, loss_func, epoch, optimizer, device):
    model.eval()
    val_losses = 0.0
    num_batches = len(validation_loader)
    for batch in validation_loader:
        input_img, _ = batch
        input_img = input_img.to(device)
        with torch.no_grad():
            output_img, stim_intensity = model(input_img)
            losses_dict = loss_func(output_img, input_img, model.parameters(), epoch, optimizer, mode="eval")
            loss = sum(losses_dict.values())
            val_losses += loss
    val_loss = val_losses / num_batches
    return val_loss, torch.sum(stim_intensity > 0).item()


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


if __name__ == "__main__":
    args = model_config.parse_arguments()
    final_config = vars(args)
    try:
        weights, training_data = train_model(args)
        torch.save(weights, f"{args.output_dir}/model.pth")
        torch.save(training_data, f"{args.output_dir}/final_training_data.pth")
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
    np.save(f"{args.output_dir}/model_config.npy", final_config)
