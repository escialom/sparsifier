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
from model2 import PhospheneOptimizer
from clipasso.models.loss import Loss
from clipasso import painterly_rendering

torch.autograd.set_detect_anomaly(True)

def train_model(args):

    # Init model, optimizer and loss
    model = PhospheneOptimizer(args=args,
                               simulator_params=dynaphos.utils.load_params("./config/config_dynaphos/params.yaml"),
                               electrode_grid=1024)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = Loss(args)

    # Prepare dataloaders
    transform = transforms.Compose([transforms.Resize((args.image_scale, args.image_scale)),
                                    transforms.CenterCrop(args.image_scale),
                                    transforms.ToTensor()])
    train_dataset = ImageFolder(root=args.train_set, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) #args.batch_size
    val_dataset = ImageFolder(root=args.val_set)

    # Prepare training loop
    epoch_loss = {}
    training_data = {}
    validation_data = {}
    if args.display:
        epoch_range = range(args.num_iter)
    else:
        epoch_range = tqdm(range(args.num_iter))

    # Training loop
    for epoch in epoch_range:
        for batch in train_loader:
            inuput_imgs,_ = batch
            inuput_imgs = inuput_imgs.to(args.device)
            optimizer.zero_grad()
            output_imgs = model(inuput_imgs)
            losses_dict = loss_func(output_imgs, inuput_imgs, model.parameters(), epoch, optimizer)
            loss = sum(losses_dict.values())
            loss.backward()
            optimizer.step()
        # Save loss of each epoch
        epoch_loss[epoch] = {'loss': loss}

        # Display epoch loss during training
        print(f'Epoch [{epoch}/{len(epoch_range)}], Loss: {loss.item():.6f}')

        # Save ongoing training data every N epochs (defined in args.save_interval)
        if epoch > 0 and epoch % args.save_interval == 0:
            # Gather current training data to the aggregated dictionary
            training_data[epoch] = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch_loss': epoch_loss[epoch].get('loss')
            }
            # TODO: Change this part
            #save_img(output_imgs, os.path.join(args.output_dir, "optimized_img.png"))

        # Validation step every N epochs (defined in args.eval_interval)
        # if epoch > 0 and epoch % args.eval_interval == 0:
        #     avg_val_loss = validate_model(model, val_dataset, loss_func, epoch, optimizer, args)
        #     # Gather current training data to the aggregated dictionary
        #     validation_data[epoch] = {
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'val_loss': avg_val_loss
        #     }

    # Save the data gathered during training
    filepath_data = os.path.join(args.output_dir, "training_data_checkpoints.pth")
    torch.save(training_data, filepath_data)
    # Get training data of the last epoch
    training_data_last_epoch = {
        'epoch_loss': epoch_loss,
        'val_loss': validation_data.get('val_loss'),
        'final_epoch': args.num_iter-1,
        'args': args
    }
    # Get the optimized image
    # save_img(output_imgs, os.path.join(args.output_dir, "optimized_img.png"))
    save_images(output_imgs, save_prefix='output_image')


    return model.state_dict(), training_data_last_epoch


def validate_model(model, validation_loader, loss_func, epoch, optimizer, args):
    # When setting torch.no_grad(), there is a bug in clipasso preventing us to interface with clip in the forward
    # loop of the model. Therefore, the model stays in training mode but the calculated losses are not backpropagated.
    val_losses = []
    for val_sample in validation_loader:
        input_img, _ = val_sample
        input_img, mask = clipasso.painterly_rendering.get_target(args, input_img)
        input_img = input_img.to(args.device)
        optimizer.zero_grad()
        output_img = model(input_img)
        output_img = output_img.to(args.device)
        losses_dict = loss_func(output_img, input_img, model.parameters(), epoch, mode="eval")
        val_loss = sum(losses_dict.values())
        val_losses.append(val_loss.item())
    avg_val_loss = sum(val_losses) / len(val_losses)
    return avg_val_loss



# def save_img(output_img, file_name_img="output_img.png"):
#     img = output_img.squeeze(0)
#     img = transforms.ToPILImage()(img)
#     img.save(file_name_img)

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
