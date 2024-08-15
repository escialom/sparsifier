import torch
import traceback
import sys

import numpy as np
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from config import model_config
import clipasso
import dynaphos
from model import PhospheneOptimizer
from clipasso.models.loss import Loss
from clipasso import painterly_rendering

torch.autograd.set_detect_anomaly(True)

def main(model_params):

    model = PhospheneOptimizer(model_params=model_params,
                               simulator_params=dynaphos.utils.load_params("./config/config_dynaphos/params.yaml"),
                               electrode_grid=1024)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params.lr)
    loss_func = Loss(model_params)
    train_dataset = ImageFolder(root=model_params.train_set)
    val_dataset = ImageFolder(root=model_params.val_set)

    epoch_loss = []
    val_loss = []
    if model_params.display:
        epoch_range = range(model_params.num_iter)
    else:
        epoch_range = tqdm(range(model_params.num_iter))

    for epoch in epoch_range:
        sample_losses = []

        for sample in train_dataset:
            input_img, _ = sample
            # Data preprocessing and augmentation performed by clipasso from PIL images
            input_img, mask = clipasso.painterly_rendering.get_target(model_params, input_img)
            input_img = input_img.to(model_params.device)
            optimizer.zero_grad()
            output_img = model(input_img)
            losses_dict = loss_func(output_img, input_img, model.parameters(), epoch, optimizer)
            loss = sum(losses_dict.values())
            loss.backward()
            optimizer.step()
            sample_losses.append(loss.item())

        # Average loss over the epoch
        avg_epoch_loss = sum(sample_losses) / len(sample_losses)
        epoch_loss.append({
            "epoch": epoch,
            "epoch_loss": avg_epoch_loss
        })

        # Validation step every N epochs (defined in model_params.eval_interval)
        if model_params.eval_interval > 0 and epoch % model_params.eval_interval == 0:
            avg_val_loss = validate_model(model, val_dataset, loss_func, model_params)
            val_loss.append({
                "epoch": epoch,
                "val_loss": avg_val_loss
            })

        # Save ongoing training data every N epochs (defined in model_params.save_interval)
        if model_params.save_interval > 0 and epoch % model_params.eval_interval == 0:
            checkpoint_file = f"checkpoint_epoch_{epoch}.pth"
            save_data(model, optimizer, epoch, epoch_loss, val_loss, model_params, file_name=checkpoint_file)

    # Save the final model # TODO: save model outside of the training loop
    final_model_file = "model.pth"
    torch.save(model.state_dict(), final_model_file)

    # Save the final training data
    final_training_data = {
        'epoch_loss': epoch_loss,
        'val_loss': val_loss,
        'final_epoch': model_params.num_iter,
        'model_params': model_params
    }
    torch.save(final_training_data, "final_training_data.pth") # TODO: save training data outside of the training loop

    return model.state_dict(), final_training_data


# TODO: check if this is really using requires_grad = False for the saliency map inside the model
# TODO: check that the loss_func is called correctly (because of arguments)
def validate_model(model, validation_loader, loss_func, model_params):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for val_sample in validation_loader:
            input_img, _ = val_sample
            input_img = input_img.to(model_params.device)
            output_img = model(input_img)
            losses_dict = loss_func(output_img, input_img, model.parameters())
            val_loss = sum(losses_dict.values())
            val_losses.append(val_loss.item())
    avg_val_loss = sum(val_losses) / len(val_losses)
    model.train()
    return avg_val_loss


def save_data(model, optimizer, epoch, epoch_loss, val_loss, model_params, file_name="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch_loss': epoch_loss,
        'val_loss': val_loss,
        'model_params': model_params
    }
    # TODO: Save optimized image
    torch.save(checkpoint, file_name)


if __name__ == "__main__":
    model_params = model_config.parse_arguments()
    final_config = vars(model_params)
    try:
        configs_to_save = main(model_params)
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
    for k in configs_to_save.keys():
        final_config[k] = configs_to_save[k]
    np.save(f"{model_params.output_dir}/config.npy", final_config)
