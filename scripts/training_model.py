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
from src import utils
from config import model_config
from src.model import PhospheneOptimizer
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
                               batch_size=args.batch_size_training,
                               phos_density=args.phos_density)
    model.to(args.device)
    loss_func = Loss(args)
    best_loss = float("inf")

    # Prepare optimizer, warmup and cos decay schedule
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_lambda = utils.make_lr_lambda(args.warmup_duration)
    scheduler_warmup = LambdaLR(optimizer, lr_lambda=lr_lambda)
    if args.lr_scheduler:
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=args.num_iter)

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
    # Prepare plot for validation losses each N steps
    n_steps_list = []
    fig_val_step, ax_val_step = plt.subplots()
    plt.show(block=False)
    val_losses_steps = []
    ax_val_step.set_xlabel("Validation step")
    ax_val_step.set_ylabel("Validation loss")
    data_tracked_val_imgs = {}

    # Prepare training loop
    epoch_loss_dict = {}
    training_data = {}
    val_loss_dict = {}
    validation_data = {}
    epoch_range = tqdm(range(args.num_iter))
    n_steps = 0
    patience_counter = 0

    # Training loop
    model.train()
    for epoch in epoch_range:
        similarity_dicts = []
        epoch_loss = 0.0
        num_batches = len(train_loader)
        for batch in train_loader:
            n_steps += 1
            input_imgs, _ = batch
            input_imgs = input_imgs.to(args.device)
            optimizer.zero_grad()
            output_imgs, stim_intensity, _, _ = model(input_imgs)
            # Mask input and output images to get black backgrounds for the clipasso loss calculation
            masked_input_imgs, mask = mask_input_imgs(input_imgs)
            masked_output_imgs = utils.mask_imgs(output_imgs, mask)
            losses_dict, similarity_dict = loss_func(masked_output_imgs,
                                                     masked_input_imgs,
                                                     model.parameters(),
                                                     epoch,
                                                     optimizer,
                                                     mode = "train")
            similarity_dicts.append(similarity_dict)
            clipasso_loss = sum(losses_dict.values())
            # Penalize background activation (model should focus on foreground)
            background_activations = output_imgs * (1 - mask)
            background_penalization_term = torch.mean(background_activations ** 2)
            # Final loss equation
            loss = clipasso_loss + args.penalization_weight * background_penalization_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            # Validate model every N steps
            if n_steps % args.val_step == 0:
                n_steps_list.append(n_steps)
                val_loss_batches = validate_model(model, mask_input_imgs, val_loader, loss_func, epoch, optimizer,
                                                  args.device)
                val_losses_steps.append(val_loss_batches)
                ax_val_step.cla()
                ax_val_step.plot(n_steps_list, val_losses_steps, label='Validation Loss', color='b')
                plt.draw()
                fig_val_step.canvas.flush_events()
                plt.pause(0.1)
                # Track validation image
                utils.track_val_imgs(args,
                                     n_img_tracked=args.num_img_tracked,
                                     dataloader=val_loader,
                                     model=model,
                                     data_tracked_val_imgs=data_tracked_val_imgs,
                                     n_steps=n_steps,
                                     device='cpu',
                                     seed=args.seed)
        epoch_loss = epoch_loss / num_batches
        epoch_loss_dict[epoch] = {'loss': epoch_loss}
        mean_scores = { # TODO: Rename this
            layer: sum(d[layer] for d in similarity_dicts) / len(similarity_dicts)
            for layer in similarity_dicts[0]  # iterate over keys
        }
        print(mean_scores)

        # Apply lr warmup/decay
        if epoch < args.warmup_duration:
            scheduler_warmup.step()
        elif args.lr_scheduler:
            scheduler_cosine.step()
        lr = optimizer.param_groups[0]['lr']

        # Save ongoing training data every epoch
        training_data[epoch] = {'epoch_loss': epoch_loss_dict[epoch].get('loss')}
        print(f'Epoch [{epoch}/{len(epoch_range)}], Training Loss: {epoch_loss:.5f}')

        # Save ongoing validation data every epoch
        val_loss = validate_model(model, mask_input_imgs, val_loader, loss_func, epoch, optimizer, args.device)
        val_loss_dict[epoch] = {'loss': val_loss}
        # Store the current validation state, model, and optimizer info
        validation_data[epoch] = {'val_loss': val_loss_dict[epoch].get('loss')}
        print(f"Fc cos loss = {losses_dict['fc'] / args.clip_fc_loss_weight: .5f}")
        print(f'Epoch [{epoch}/{len(epoch_range)}], Validation Loss: {val_loss:.5f}')

        # Update files
        torch.save(training_data, os.path.join(args.output_path, "training_data_checkpoints.pth"))
        torch.save(validation_data, os.path.join(args.output_path, "validation_data_checkpoints.pth"))

        # Save model state each time we get a better val loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': val_loss_dict[epoch].get('loss')},
                       os.path.join(args.output_path, f"checkpoint_epoch_{epoch}.pth"))

        # Early stopping: difference between 2 val loss checks must be smaller than threshold
        if epoch >= args.epoch_check and epoch % args.epoch_check == 0:
            convergence = abs(val_loss-val_loss_dict[epoch-args.epoch_check]['loss']) <= args.delta_val_losses
            if convergence:
                patience_counter += 1
                if patience_counter == args.patience:
                    torch.save({'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'train_loss': val_loss_dict[epoch].get('loss')},
                               os.path.join(args.output_path, f"model_converged_epoch_{epoch}.pth"))
                    print(f"Early stopping at epoch {epoch}. ")
                    break
            else:
                patience_counter = 0

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

    return model.state_dict()


def validate_model(model, mask_input_imgs, validation_loader, loss_func, epoch, optimizer, device):
    model.eval()
    val_losses = 0.0
    num_batches = len(validation_loader)
    for batch in validation_loader:
        input_imgs, _ = batch
        input_imgs = input_imgs.to(device)
        with torch.no_grad():
            output_imgs, _, _ = model(input_imgs)
            # Mask input and output images to get black background for the clipasso loss calculation
            masked_input_imgs, mask = mask_input_imgs(input_imgs)
            masked_output_imgs = utils.mask_imgs(output_imgs, mask)
            losses_dict, _ = loss_func(masked_output_imgs, masked_input_imgs, model.parameters(), epoch, optimizer,
                                    mode="eval")
            clipasso_loss = sum(losses_dict.values())
            # Get the background activations to be penalized in the loss (model should focus on foreground)
            background_activations = output_imgs * (1 - mask)
            background_penalization_term = torch.mean(background_activations ** 2)
            # Get the final loss equation
            loss = clipasso_loss + args.penalization_weight * background_penalization_term
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
        torch.save(model, f"{args.output_path}/model.pth")
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
    np.save(f"{args.output_path}/model_config.npy", final_config)
