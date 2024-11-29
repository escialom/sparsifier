import copy
import os
import sys
import torch
import traceback

import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import dynaphos
import utils
from config import model_config
from model import PhospheneOptimizer
from clipasso.models.loss import Loss


def train_model(args):

    # Prepare dataloaders with augmentations
    augmentations = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.RandomVerticalFlip(0.5),
                                        transforms.RandomRotation(degrees=(-15, 15)),
                                        transforms.ColorJitter(brightness=0.3,
                                                               contrast=0.3,
                                                               saturation=0.3,
                                                               hue=0.1)])
    train_dataset = ImageFolder(root=args.train_set, transform=augmentations)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_training, shuffle=True)
    val_dataset = ImageFolder(root=args.val_set, transform=transforms.ToTensor())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_validation, shuffle=True)

    # Init class for segmenting images
    mask_input_imgs = utils.MaskImgs(args)

    # Init model and clipasso loss
    model = PhospheneOptimizer(args=args,
                               simulator_params=dynaphos.utils.load_params("./config/config_dynaphos/params.yaml"),
                               electrode_grid=1024,
                               batch_size=args.batch_size_training)
    model.to(args.device)
    loss_func = Loss(args)

    # Prepare optimizer: warmup and cos decay schedule
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_lambda = utils.make_lr_lambda(args.warmup_duration)
    scheduler_warmup = LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=args.num_iter)

    # Init class for checking model convergence (early stopping criterion)
    # model_converged = utils.EarlyStopping(patience=args.check_interval,
    #                                       window_size=args.check_interval,
    #                                       min_delta=args.min_val_loss_diff,
    #                                       verbose=True)

    # Randomly select training and validation images
    dir_train_img_og = os.path.join(args.output_dir, "train_img_og")
    dir_val_img_og = os.path.join(args.output_dir, "val_img_og")
    utils.copy_random_images_per_class(source_dir=args.train_set,
                                       destination_dir=dir_train_img_og,
                                       num_images_per_class=10)
    utils.copy_random_images_per_class(source_dir=args.val_set,
                                       destination_dir=dir_val_img_og,
                                       num_images_per_class=10)

    # Generate output images at model init
    tracked_train_dataset = ImageFolder(root=dir_train_img_og, transform=transforms.ToTensor())
    tracked_train_loader = DataLoader(tracked_train_dataset, batch_size=1, shuffle=True)
    tracked_val_dataset = ImageFolder(root=dir_val_img_og, transform=transforms.ToTensor())
    tracked_val_loader = DataLoader(tracked_val_dataset, batch_size=1, shuffle=False)
    utils.track_images(args,
                       model.eval(),
                       tracked_train_dataset,
                       tracked_train_loader,
                       input_dir=dir_train_img_og,
                       output_dir=os.path.join(args.output_dir, "train_img_tracking"),
                       at_init=True)
    utils.track_images(args,
                       model.eval(),
                       tracked_val_dataset,
                       tracked_val_loader,
                       input_dir=dir_val_img_og,
                       output_dir=os.path.join(args.output_dir, "val_img_tracking"),
                       at_init=True)

    # Prepare loss plots
    plt.ion()
    fig, ax = plt.subplots()
    plt.show(block=False)
    epochs = []
    val_losses = []
    epoch_losses = []
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    # Prepare learning rate plot
    fig_lr, ax_lr = plt.subplots()
    plt.show(block=False)
    learning_rates = []
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("Learning Rate")

    # Prepare training loop
    epoch_loss_dict = {}
    training_data = {}
    val_loss_dict = {}
    validation_data = {}
    prev_val_loss_checked = None
    epoch_range = tqdm(range(args.num_iter))

    # Training loop
    model.train()
    for epoch in epoch_range:
        epoch_loss = 0.0
        num_batches = len(train_loader)
        for batch in train_loader:
            input_imgs, _ = batch
            input_imgs = input_imgs.to(args.device)
            optimizer.zero_grad()
            output_imgs, stim_intensity = model(input_imgs)
            # Mask input and output images to get black backgrounds for the clipasso loss calculation
            masked_input_imgs, mask = mask_input_imgs(input_imgs)
            masked_output_imgs = utils.mask_imgs(output_imgs, mask)
            losses_dict = loss_func(masked_output_imgs, masked_input_imgs, model.parameters(), epoch, optimizer, mode = "train")
            clipasso_loss = sum(losses_dict.values())
            # Get the background activations to be penalized in the loss (model should focus on foreground)
            background_activations = output_imgs * (1 - mask)
            background_penalization_term = torch.mean(background_activations ** 2)
            loss = clipasso_loss + args.penalization_weight * background_penalization_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / num_batches
        epoch_loss_dict[epoch] = {'loss': epoch_loss}

        # Apply lr warmup/decay
        if epoch < args.warmup_duration:
            scheduler_warmup.step()
        else:
            scheduler_cosine.step()
        lr = optimizer.param_groups[0]['lr']

        # Save ongoing training data every epoch
        training_data[epoch] = {
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
            'epoch_loss': epoch_loss_dict[epoch].get('loss')
        }
        print(f'Epoch [{epoch}/{len(epoch_range)}], Training Loss: {epoch_loss:.5f}')

        # Save ongoing validation data every epoch
        val_loss = validate_model(model, mask_input_imgs, val_loader, loss_func, epoch, optimizer, args.device)
        val_loss_dict[epoch] = {'loss': val_loss}
        # Store the current validation state, model, and optimizer info
        validation_data[epoch] = {
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
            'val_loss': val_loss_dict[epoch].get('loss')
        }
        print(f'Epoch [{epoch}/{len(epoch_range)}], Validation Loss: {val_loss:.5f}')

        # Every N epochs (defined in args.check_interval), save tracked validation image and check for early stopping
        if epoch >= 0 and epoch % args.check_interval == 0:
            # Save the optimized (tracked) training/validation images and register its metadata
            model.eval()
            utils.track_images(args,
                               model,
                               tracked_train_dataset,
                               tracked_train_loader,
                               input_dir=dir_train_img_og,
                               output_dir=os.path.join(args.output_dir, "train_img_tracking"),
                               epoch=epoch)
            utils.track_images(args,
                               model,
                               tracked_val_dataset,
                               tracked_val_loader,
                               input_dir=dir_val_img_og,
                               output_dir=os.path.join(args.output_dir, "val_img_tracking"),
                               epoch=epoch)
            model.train()

            # Check if convergence criterion is met
            if prev_val_loss_checked is not None:
                val_loss_diff = abs(val_loss - prev_val_loss_checked)
                if val_loss_diff < args.min_val_loss_diff:
                    # Update files before breaking the training
                    torch.save(training_data, os.path.join(args.output_dir, "training_data_checkpoints.pth"))
                    torch.save(validation_data, os.path.join(args.output_dir, "validation_data_checkpoints.pth"))
                    #torch.save(data_tracked_val_imgs, os.path.join(args.output_dir, "data_tracked_validation_img.pth"))
                    print(f"Convergence criterion met at epoch {epoch}.")
            prev_val_loss_checked = val_loss

        # Update files
        torch.save(training_data, os.path.join(args.output_dir, "training_data_checkpoints.pth"))
        torch.save(validation_data, os.path.join(args.output_dir, "validation_data_checkpoints.pth"))
        #torch.save(data_tracked_val_imgs, os.path.join(args.output_dir, "data_tracked_validation_img.pth"))

        # Update variables for plotting
        epochs.append(epoch)
        val_losses.append(val_loss_dict[epoch]['loss'])
        epoch_losses.append(epoch_loss_dict[epoch]['loss'])
        learning_rates.append(lr)

        # Clear and update the loss plot
        ax.cla()
        ax.plot(epochs, val_losses, label='Validation Loss', color='b')
        ax.plot(epochs, epoch_losses, label='Training Loss', color='r')
        ax.legend()
        plt.draw()
        fig.canvas.flush_events()

        # Clear and update the learning rate plot
        ax_lr.cla()
        ax_lr.plot(epochs, learning_rates, label='Learning Rate', color='g')
        ax_lr.legend()
        plt.draw()
        fig_lr.canvas.flush_events()

        plt.pause(0.1)

    plt.ioff()
    #plt.show()

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
            utils.save_images(output_img, output_img_dir, input_filename, epoch)
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


def validate_model(model, mask_input_imgs, validation_loader, loss_func, epoch, optimizer, device):
    model.eval()
    val_losses = 0.0
    num_batches = len(validation_loader)
    for batch in validation_loader:
        input_imgs, _ = batch
        input_imgs = input_imgs.to(device)
        with torch.no_grad():
            output_imgs, _ = model(input_imgs)
            # Mask input and output images to get black background for the clipasso loss calculation
            masked_input_imgs, mask = mask_input_imgs(input_imgs)
            masked_output_imgs = utils.mask_imgs(output_imgs, mask)
            losses_dict = loss_func(masked_output_imgs, masked_input_imgs, model.parameters(), epoch, optimizer,
                                    mode="eval")
            clipasso_loss = sum(losses_dict.values())
            # Get the background activations to be penalized in the loss (model should focus on foreground)
            background_activations = output_imgs * (1 - mask)
            background_penalization_term = torch.mean(background_activations ** 2)
            loss = clipasso_loss + 1 * background_penalization_term
            val_losses += loss.item()
    # Get the average validation loss of current epoch
    val_loss_epoch = val_losses / num_batches
    # Set model back to training mode
    model.train()
    return val_loss_epoch


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
