import torch
import traceback
import sys

import numpy as np
from torchvision import transforms
import torch.nn.utils as utils
from torch.utils.data import DataLoader
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
    train_dataset = ImageFolder(root=model_params.train_set, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=model_params.batch_size, shuffle=True)
    to_pil = transforms.ToPILImage()

    epoch_loss = []
    if model_params.display:
        epoch_range = range(model_params.num_iter)
    else:
        epoch_range = tqdm(range(model_params.num_iter))
    for epoch in epoch_range:
        counter = 0
        for batch in train_loader:
            if counter > 0:
                break
            losses = []
            batch_losses = []
            # we loop over samples in the batch as clipasso expect only one sample as input
            images = batch[0]
            for sample in images:
                # Data preprocessing and augmentation performed by clipasso from PIL images
                input_img = to_pil(sample.squeeze(0))
                input_img, mask = clipasso.painterly_rendering.get_target(model_params, input_img)
                # input_img[input_img == 0.] = 1. # It was because of Xdog which would produce weird saliency maps
                input_img = input_img.to(model_params.device)
                optimizer.zero_grad()
                output_img = model(input_img)
                # output_img[output_img == 1.] = 0.  # Turn it back to black
                losses_dict = loss_func(output_img, input_img, model.parameters(), epoch, optimizer)
                losses.append(sum(losses_dict.values()))
            counter =+ 1
            # Average loss over samples in the batch
            loss = sum(losses) / len(losses)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        # Save loss of each epoch
        epoch_loss.append({
            "epoch": epoch + 1,
            "epoch_loss": sum(batch_losses) / len(batch_losses)
        })
    return epoch_loss


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
