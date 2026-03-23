import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

import dynaphos
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator
from src.ContourExtract import ContourExtract


class MiniConvNet(nn.Module):
    def __init__(self, args, seed=None):
        super(MiniConvNet, self).__init__()
        self.size = args.image_scale

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.use_deterministic_algorithms(True)

        self.downsampling = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU()
        )

        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        xs = self.downsampling(x)
        torch.use_deterministic_algorithms(False)
        theta = self.upsampling(xs)
        theta = torch.clamp(theta, min=theta.mean(), max=theta.max())
        theta = torch.sigmoid(theta)
        return theta


class PhospheneOptimizer(nn.Module):
    def __init__(self, args,
                 simulator_params,
                 electrode_grid,
                 batch_size,
                 phos_density):
        super(PhospheneOptimizer, self).__init__()

        self.args = args
        self.simulator_params = simulator_params
        self.electrode_grid = electrode_grid
        self.batch_size = batch_size
        self.phos_density = phos_density
        self.phosphene_coords = dynaphos.cortex_models.get_visual_field_coordinates_probabilistically(self.simulator_params, self.electrode_grid, use_seed=True)
        self.get_learnable_params = MiniConvNet(self.args, seed=args.seed).to(args.device)
        self.gray_scale = transforms.Grayscale(num_output_channels=1)
        self.padding = transforms.Pad(padding=args.padding_pix*args.sigma_kernel, fill=args.padding_color)
        self.cropping = transforms.CenterCrop(args.image_scale)
        self.get_contours = ContourExtract(n_orientations=args.n_orientations, sigma_kernel=args.sigma_kernel, lambda_kernel=args.lambda_kernel)
        self.contours_batch = torch.empty(self.batch_size, 1, self.args.image_scale, self.args.image_scale)
        self.simulator = PhospheneSimulator(self.simulator_params, self.phosphene_coords, batch_size=self.batch_size, phos_density=self.phos_density, rng=np.random.default_rng(self.simulator_params['run']['seed']))

    def forward(self, img_batch):
        # The first dimension of img_batch can change during the validation step.
        # If so, reinit model and contours batch with correct batch size
        if self.simulator.batch_size != img_batch.shape[0]:
            self.simulator = PhospheneSimulator(self.simulator_params, self.phosphene_coords, batch_size=img_batch.shape[0], phos_density=self.phos_density, rng=np.random.default_rng(self.simulator_params['run']['seed']))
            self.contours_batch = torch.empty(img_batch.shape[0], 1, self.args.image_scale, self.args.image_scale)
        contours_batch = self.contours_extraction(img_batch, requires_grad=True)
        phosphene_placement_map = self.get_learnable_params(contours_batch)
        # Rescale pixel intensities between [0, <max_stimulation_intensity_ampere>]
        phosphene_placement_map = self.normalized_rescaling(phosphene_placement_map, max_stimulation_intensity=self.simulator_params['sampling']['stimulus_scale'])
        # Make the phosphene_placement_map as a stimulation vector for the phosphene simulator
        phosphene_placement_map = self.simulator.sample_stimulus(phosphene_placement_map)
        self.simulator.reset()
        optimized_im, intensity = self.simulator(phosphene_placement_map)
        # Get the simulated total current in Amperes for the current batch
        current = self.simulator.effective_charge_per_second
        top_elec_activated = (intensity > 0).float()
        sim_current = (current * top_elec_activated).sum().item()
        # Add the channel dimension back [Bx3xHxW]
        optimized_im = optimized_im.unsqueeze(0)
        optimized_im = optimized_im.permute(1, 0, 2, 3)
        optimized_im = optimized_im.repeat(1, 3, 1, 1)
        return optimized_im, intensity, sim_current, phosphene_placement_map

    def contours_extraction(self, img_batch, requires_grad=False):
        with torch.no_grad():
            for i in range(img_batch.shape[0]):
                img = self.gray_scale(img_batch[i])
                img = self.padding(img)
                contours = self.get_contours(img.unsqueeze(0))
                # Collapse the contours extracted in each orientation
                contours = self.get_contours.create_collapse(contours)
                # Remove padding to get back to dimensions of input img
                contours = self.cropping(contours)
                self.contours_batch[i] = contours
            # Normalize each image of the batch between 0 and 1
            self.contours_batch = ((self.contours_batch - self.contours_batch.amin(dim=(1, 2, 3), keepdim=True)) /
                                   (self.contours_batch.amax(dim=(1, 2, 3), keepdim=True) -
                                    self.contours_batch.amin(dim=(1, 2, 3), keepdim=True) + 1e-8))
            self.contours_batch.requires_grad = requires_grad
        return self.contours_batch.to(self.args.device)

    def normalized_rescaling(self, phosphene_placement_map, max_stimulation_intensity=1):
        """Normalize <img> and rescale the pixel intensities in the range [0, <stimulus_scale>].
        <stimulus_scale> is defined in the parameter file.
        The output image represents the stimulation intensity map.
        return: image with rescaled pixel values (stimulation intensity map in Ampères)."""

        img_norm = ((phosphene_placement_map - phosphene_placement_map.min()) /
                    (phosphene_placement_map.max() - phosphene_placement_map.min()))
        return img_norm * max_stimulation_intensity

