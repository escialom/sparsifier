import numpy as np
import torch
import torch.nn as nn
from clipasso.models import painter_params as clipasso_model
import dynaphos
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator


class InitMap(torch.nn.Module):
    def __init__(self, args):
        super(InitMap, self).__init__()
        self.args = args
        self.device = self.args.device
        self.clipasso_model = clipasso_model.Painter(self.args, device=self.device)
        self.contour_extractor = clipasso_model.XDoG_()
        self.image_scale = (self.args.image_scale, self.args.image_scale)
        self.init_map_soft_batch = torch.empty((1, 1, *self.image_scale), device=self.device)

    def forward(self, img_batch, requires_grad):
        batch_size = img_batch.shape[0]
        if self.init_map_soft_batch.shape[0] != batch_size:
            self.init_map_soft_batch = torch.empty((batch_size, 1, *self.image_scale), device=self.device)

        # Extract contours for each image of the batch
        with torch.no_grad():
            for i in range(img_batch.shape[0]):
                image = img_batch[i].permute(1, 2, 0).cpu().numpy()
                image_contours = self.contour_extractor(image, k=10)
                image_contours = (1-image_contours)
                # Softmax the contour image
                init_map_soft = torch.tensor(image_contours, device=self.device)
                init_map_soft[image_contours > 0] = self.softmax(image_contours[image_contours > 0], tau=self.args.softmax_temp)
                init_map_soft /= init_map_soft.max()
                self.init_map_soft_batch[i] = init_map_soft.detach()
        self.init_map_soft_batch.requires_grad = requires_grad

        return self.init_map_soft_batch

    def softmax(self, x, tau=0.2):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=self.device)
        e_x = torch.exp(x / tau)
        return e_x / e_x.sum()


class MiniConvNet(nn.Module):
    def __init__(self, args, seed=None):
        super(MiniConvNet, self).__init__()
        self.size = args.image_scale

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.use_deterministic_algorithms(True)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, 3, stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.conv_padding = nn.Sequential(
            nn.Upsample((args.image_scale, args.image_scale), mode='nearest'))

    def forward(self, x):
        xs = self.localization(x)
        torch.use_deterministic_algorithms(False)
        theta = self.conv_padding(xs)
        theta = torch.clamp(theta, min=theta.mean(), max=theta.max())
        theta = torch.sigmoid(theta)
        return theta


class PhospheneOptimizer(nn.Module):
    def __init__(self, args,
                 simulator_params,
                 electrode_grid,
                 batch_size):
        super(PhospheneOptimizer, self).__init__()

        self.args = args
        self.simulator_params = simulator_params
        self.electrode_grid = electrode_grid
        self.batch_size = batch_size
        self.phosphene_coords = dynaphos.cortex_models.get_visual_field_coordinates_probabilistically(self.simulator_params, self.electrode_grid, use_seed=True)
        self.get_learnable_params = MiniConvNet(self.args,seed=args.seed).to(args.device)
        self.get_contours = InitMap(self.args)
        self.simulator = PhospheneSimulator(self.simulator_params, self.phosphene_coords, batch_size=self.batch_size)

    def forward(self, img_batch):
        # The first dimension of img_batch can change during the validation step.
        # If so, reinit model with correct batch size
        if self.simulator.batch_size != img_batch.shape[0]:
            self.simulator = PhospheneSimulator(self.simulator_params, self.phosphene_coords, batch_size=img_batch.shape[0])
        contours = self.get_contours(img_batch, requires_grad=True)
        phosphene_placement_map = self.get_learnable_params(contours)
        # Rescale pixel intensities between [0, <max_stimulation_intensity_ampere>]
        phosphene_placement_map = self.normalized_rescaling(phosphene_placement_map, max_stimulation_intensity=self.simulator_params['sampling']['stimulus_scale'])
        # Make the phosphene_placement_map as a stimulation vector for the phosphene simulator
        phosphene_placement_map = self.simulator.sample_stimulus(phosphene_placement_map)
        self.simulator.reset()
        optimized_im, stim_intensity = self.simulator(phosphene_placement_map)
        # Add the channel dimension back [Bx3xHxW]
        optimized_im = optimized_im.unsqueeze(0)
        optimized_im = optimized_im.permute(1, 0, 2, 3)
        optimized_im = optimized_im.repeat(1, 3, 1, 1)
        # detach and delete variables to save memory
        contours = contours.detach()
        del contours
        phosphene_placement_map = phosphene_placement_map.detach()
        del phosphene_placement_map
        return optimized_im, stim_intensity

    def normalized_rescaling(self, phosphene_placement_map, max_stimulation_intensity=1):
        """Normalize <img> and rescale the pixel intensities in the range [0, <stimulus_scale>].
        <stimulus_scale> is defined in the parameter file.
        The output image represents the stimulation intensity map.
        return: image with rescaled pixel values (stimulation intensity map in Ampères)."""

        img_norm = (phosphene_placement_map - phosphene_placement_map.min()) / (phosphene_placement_map.max() - phosphene_placement_map.min())
        return img_norm * max_stimulation_intensity

