import PIL
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from models.loss import Loss
from simplified_painter import Phosphene_model
import config
import sketch_utils as s_utils
import os
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import dynaphos
from dynaphos import utils, cortex_models
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator


# Your PhospheneTransformerNet class goes here
class PhospheneTransformerNet(nn.Module):
    def __init__(self, size):
        super(PhospheneTransformerNet, self).__init__()
        self.size = size
        # Define each layer separately to allow for intermediate checks
        self.conv1 = nn.Conv2d(1, 8, kernel_size=7)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(8, 10, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.relu2 = nn.ReLU(True)
        # Linear layer
        self.fc_loc = nn.Linear(10 * 52 * 52, 1024)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize Convolutional layers with Kaiming normalization
        for conv in [self.conv1, self.conv2]:
            init.kaiming_uniform(conv.weight, mode='fan_out', nonlinearity='relu')
            if conv.bias is not None:
                init.constant_(conv.bias, 0)

        # Initialize Fully Connected layer with Xavier initialization
        init.xavier_normal_(self.fc_loc.weight)
        init.constant_(self.fc_loc.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        assert not torch.isnan(x).any(), 'NaN detected after conv1'

        x = self.pool1(x)
        assert not torch.isnan(x).any(), 'NaN detected after pool1'

        x = self.relu1(x)
        assert not torch.isnan(x).any(), 'NaN detected after relu1'

        x = self.conv2(x)
        assert not torch.isnan(x).any(), 'NaN detected after conv2'

        x = self.pool2(x)
        assert not torch.isnan(x).any(), 'NaN detected after pool2'

        x = self.relu2(x)
        assert not torch.isnan(x).any(), 'NaN detected after relu2'

        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)
        assert not torch.isnan(x).any(), 'NaN detected after flattening'

        x = self.fc_loc(x)
        assert not torch.isnan(x).any(), 'NaN detected after fc_loc'

        return x


def get_target_and_mask(args):
    target = Image.open(args.target)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    masked_im, mask = s_utils.get_mask_u2net(args, target)
    if args.mask_object:
        target = masked_im
    if args.fix_scale:
        target = s_utils.fix_image_scale(target)

    transforms_ = []
    if target.size[0] != target.size[1]:
        transforms_.append(transforms.Resize(
            (args.image_scale, args.image_scale), interpolation=PIL.Image.BICUBIC))
    else:
        transforms_.append(transforms.Resize(
            args.image_scale, interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(args.image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    target_im = data_transforms(target).unsqueeze(0).to(args.device)

    return target_im, mask


def normalized_rescaling(img, stimulus_scale=100.e-6):
    """Normalize <img> and rescale the pixel intensities in the range [0, <stimulus_scale>].
    The output image represents the stimulation intensity map.
    param stimulus_scale: the stimulation amplitude corresponding to the highest-valued pixel.
    return: image with rescaled pixel values (stimulation intensity map in Ampères)."""

    img_norm = (img - img.min()) / (img.max() - img.min())
    return img_norm * stimulus_scale

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_stn(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the STN module
    stn = PhospheneTransformerNet(size=224).to(device)

    # Simplistic input tensor, simulating the clip_saliency_map
    input_tensor = torch.randn(1, 1, 224, 224, requires_grad=True).to(device)

    optimizer = optim.Adam(stn.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print(count_trainable_parameters(stn)) # Same amount as in the main_training.py

    # Target, for the sake of having a loss computation
    # target = torch.randn(1, 1024).to(device)

    target, mask = get_target_and_mask(args)


    params = utils.load_params(
        'C:/Users/vanholk/sparsifier/dynaphos/config/params.yaml')
    num_phosphenes = 1000

    phosphene_coords = cortex_models.get_visual_field_coordinates_probabilistically(params,
                                                                                         num_phosphenes)
    simulator = PhospheneSimulator(params, phosphene_coords)

    # Lets replace the target with our own target

    for epoch in range(100):
        optimizer.zero_grad()
        output = stn(input_tensor)
        output = normalized_rescaling(output)

        simulator.reset()

        output = simulator(output)


        output = output.unsqueeze(0)
        output = output.repeat(1, 3, 1, 1)  # Now the shape is [1, 3, H, W]
        output = output.permute(0, 1, 2, 3)  # .to(self.device)  # NHW -> NHW

        # Check output for NaN values
        assert not torch.isnan(output).any(), "Output contains NaN"

        loss = criterion(output, target)
        loss.backward()

        # Check for NaN in gradients
        for name, param in stn.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f'Gradient NaN detected in {name}'

        optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item()}')


if __name__ == "__main__":
    args = config.parse_arguments()
    train_stn(args)