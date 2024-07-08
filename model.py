import os
import numpy as np
import torch
import torch.nn as nn
import config
import clipasso.CLIP_.clip as clip
from clipasso.models import painter_params as clipasso_model
import dynaphos
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator

# Load model and phosphene simulator parameters
model_params = config.model_config.parse_arguments()
abs_path = os.path.abspath(os.getcwd())
simulator_params = dynaphos.utils.load_params(f"{abs_path}/dynaphos/config/params.yaml")


class SaliencyMap(torch.nn.Module):
    def __init__(self, args,
                 input_image = None,
                 device = None):
        super(SaliencyMap, self).__init__()

        self.args = args
        self.input_image = input_image
        self.device = device
        self.softmax_temp = args.softmax_temp

    def __call__(self):
        input_image = input_image[input_image == 0.] = 1. # background should be white
        input_image_preprocessed = clipasso_model.Painter.define_attention_input(input_image)
        clip_attention_map = clipasso_model.Painter.clip_attn()
        contour_extractor = clipasso_model.XDoG_()
        image_contours = contour_extractor(input_image_preprocessed[0].permute(1, 2, 0).cpu().numpy(), k=10)
        # Multiplication of attention map and edge map
        saliency_map = (1 - image_contours) * clip_attention_map
        saliency_map_soft = np.copy(saliency_map)
        saliency_map_soft[saliency_map > 0] = self.softmax(saliency_map[saliency_map > 0], tau=model_params.softmax_temp)
        saliency_map_soft = torch.Tensor(saliency_map_soft) / saliency_map_soft.max()
        saliency_map_soft.requires_grad = True
        return clip_attention_map, saliency_map_soft #attention_map, saliency_map

    def softmax(self, x, tau=0.2):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum()


class MiniConvNet(nn.Module):
    def __init__(self):
        super(MiniConvNet, self).__init__()
        self.localization = nn.Sequential(nn.Conv2d(1, 32, 3, stride=1, padding=0),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 64, 3, stride=2, padding=0),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 1, 3, stride=1, padding=0),
                                        nn.ReLU())
        self.conv_padding = nn.Sequential(nn.Upsample((224, 224), mode='nearest'))

    def forward(self, x):
        xs = self.localization(x)
        theta = self.conv_padding(xs)
        theta = torch.clamp(theta, min=theta.mean(), max=theta.max())
        theta = torch.sigmoid(theta)
        return theta


class PhospheneOptimizer(nn.Module):
    def __init__(self,
                 electrode_grid,
                 device=None):
        super(PhospheneOptimizer, self).__init__()

        self.model_params = model_params
        self.simulator_params = simulator_params
        self.electrode_grid = electrode_grid
        self.use_seed = True
        self.phosphene_coords = dynaphos.cortex_models.get_visual_field_coordinates_probabilistically(self.simulator_params, self.electrode_grid, self.use_seed)
        self.simulator = PhospheneSimulator(self.simulator_params, self.phosphene_coords)
        self.get_learnable_params = MiniConvNet()
        self.device = device
        self.saliency_clip_model = model_params.saliency_clip_model
        self.text_target = model_params.text_target
        self.clip_model, self.preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        self.clip_model.eval().to(self.device)

    def forward(self, input_image): #clip_attention_map, saliency_map_soft --> attention_map, saliency_map
        clip_attention_map, saliency_map = SaliencyMap(input_image)
        phosphene_placement_map = self.get_learnable_params(saliency_map.unsqueeze(0).unsqueeze(0))
        # Make the phosphene_placement_map as a stimulation vector for the phosphene simulator
        phosphene_placement_map = self.simulator.sample_stimulus(phosphene_placement_map, rescale=True)
        self.simulator.reset()
        optimized_im = self.simulator(phosphene_placement_map)
        optimized_im = optimized_im.unsqueeze(0)
        optimized_im = optimized_im.repeat(1, 3, 1, 1)
        optimized_im = optimized_im.permute(0, 1, 2, 3)
        del clip_attention_map, saliency_map
        return optimized_im