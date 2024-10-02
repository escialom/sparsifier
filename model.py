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

    def forward(self, input_image, init_mode, requires_grad):
        # Preprocessing of input image
        input_image_preprocessed = self.clipasso_model.define_attention_input(input_image)
        image_contours = self.contour_extractor(input_image_preprocessed[0].permute(1, 2, 0).cpu().numpy(), k=10)
        if init_mode == 'saliency':
            clip_attention_map = self.clipasso_model.clip_attn(input_image_preprocessed)
            # Multiplication of attention map and contours to get saliency map
            init_map = (1 - image_contours) * clip_attention_map
        elif init_mode == 'contours':
            # Just use the contours for initialization
            init_map = (1 - image_contours)
        init_map_soft = np.copy(init_map)
        init_map_soft[init_map > 0] = self.softmax(init_map[init_map > 0], tau=self.args.softmax_temp)
        init_map_soft = torch.Tensor(init_map_soft) / init_map_soft.max()
        init_map_soft.requires_grad = requires_grad

        return init_map_soft

    def softmax(self, x, tau=0.2):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum()


class MiniConvNet(nn.Module):
    def __init__(self, args, seed=None):
        super(MiniConvNet, self).__init__()
        self.size = args.image_scale

        if seed is not None:
            torch.manual_seed(seed)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, stride=1, padding=0),
            nn.ReLU()
        )

        self.conv_padding = nn.Sequential(
            nn.Upsample((args.image_scale, args.image_scale), mode='nearest'))

    def forward(self, x):
        xs = self.localization(x)
        theta = self.conv_padding(xs)
        theta = torch.clamp(theta, min=theta.mean(), max=theta.max())
        theta = torch.sigmoid(theta)
        return theta


class PhospheneOptimizer(nn.Module):
    def __init__(self, args,
                 simulator_params,
                 electrode_grid):
        super(PhospheneOptimizer, self).__init__()

        self.args = args
        self.simulator_params = simulator_params
        self.electrode_grid = electrode_grid
        self.phosphene_coords = dynaphos.cortex_models.get_visual_field_coordinates_probabilistically(self.simulator_params, self.electrode_grid, use_seed=True)
        self.simulator = PhospheneSimulator(self.simulator_params, self.phosphene_coords)
        self.get_learnable_params = MiniConvNet(self.args, seed=args.seed)
        self.get_init_map = InitMap(self.args)

    def forward(self, input_image):
        # Network is either initialized with saliency map or image's contours (defined in init_mode)
        init_map = self.get_init_map(input_image, init_mode=self.args.init_mode, requires_grad=True)
        phosphene_placement_map = self.get_learnable_params(init_map.unsqueeze(0).unsqueeze(0))
        # Rescale pixel intensities between [0, <max_stimulation_intensity_ampere>]
        phosphene_placement_map = self.normalized_rescaling(phosphene_placement_map, max_stimulation_intensity=self.simulator_params['sampling']['stimulus_scale'])
        # Make the phosphene_placement_map as a stimulation vector for the phosphene simulator
        phosphene_placement_map = self.simulator.sample_stimulus(phosphene_placement_map)
        self.simulator.reset()
        optimized_im = self.simulator(phosphene_placement_map)
        optimized_im = optimized_im.unsqueeze(0)
        optimized_im = optimized_im.repeat(1, 3, 1, 1)
        optimized_im = optimized_im.permute(0, 1, 2, 3)
        del init_map
        return optimized_im

    def normalized_rescaling(self, phosphene_placement_map, max_stimulation_intensity=1):
        """Normalize <img> and rescale the pixel intensities in the range [0, <stimulus_scale>].
        <stimulus_scale> is defined in the parameter file.
        The output image represents the stimulation intensity map.
        return: image with rescaled pixel values (stimulation intensity map in Ampères)."""

        img_norm = (phosphene_placement_map - phosphene_placement_map.min()) / (phosphene_placement_map.max() - phosphene_placement_map.min())
        return img_norm * max_stimulation_intensity

