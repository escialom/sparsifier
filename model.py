import numpy as np
import torch
import torch.nn as nn
from clipasso.models import painter_params as clipasso_model
import dynaphos
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator

# from torchvision import transforms
# import clipasso.CLIP_.clip as clip


# Load model and phosphene simulator parameters
#model_params = model_config.parse_arguments()
#abs_path = os.path.abspath(os.getcwd())
#simulator_params = dynaphos.utils.load_params(f"{abs_path}/config/config_dynaphos/params.yaml")


class SaliencyMap(torch.nn.Module):
    def __init__(self, model_params, requires_grad=True):
        super(SaliencyMap, self).__init__()
        self.model_params = model_params
        self.requires_grad = requires_grad
        self.device = self.model_params.device
        self.clipasso_model = clipasso_model.Painter(self.model_params, device=self.device)

    def forward(self, input_image):
        input_image_preprocessed = self.clipasso_model.define_attention_input(input_image)
        # Background should be white for extracting saliency map
        clip_attention_map = self.clipasso_model.clip_attn(input_image_preprocessed)
        contour_extractor = clipasso_model.XDoG_()
        image_contours = contour_extractor(input_image_preprocessed[0].permute(1, 2, 0).cpu().numpy(), k=10)
        # Multiplication of attention map and edge map
        saliency_map = (1 - image_contours) * clip_attention_map
        saliency_map_soft = np.copy(saliency_map)
        saliency_map_soft[saliency_map > 0] = self.softmax(saliency_map[saliency_map > 0], tau=self.model_params.softmax_temp)
        saliency_map_soft = torch.Tensor(saliency_map_soft) / saliency_map_soft.max()
        saliency_map_soft.requires_grad = self.requires_grad
        return clip_attention_map, saliency_map_soft #attention_map, saliency_map

    def softmax(self, x, tau=0.2):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum()


class MiniConvNet(nn.Module):
    def __init__(self, model_params):
        super(MiniConvNet, self).__init__()
        self.size = model_params.image_scale

        self.localization = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, stride=1, padding=0),
            nn.ReLU()
        )

        self.conv_padding = nn.Sequential(
            nn.Upsample((model_params.image_scale, model_params.image_scale), mode='nearest'))

    def forward(self, x):
        xs = self.localization(x)
        theta = self.conv_padding(xs)
        theta = torch.clamp(theta, min=theta.mean(), max=theta.max())
        theta = torch.sigmoid(theta)
        return theta


class PhospheneOptimizer(nn.Module):
    def __init__(self, model_params,
                 simulator_params,
                 electrode_grid):
        super(PhospheneOptimizer, self).__init__()

        self.model_params = model_params
        self.simulator_params = simulator_params
        self.electrode_grid = electrode_grid
        self.use_seed = True
        self.phosphene_coords = dynaphos.cortex_models.get_visual_field_coordinates_probabilistically(self.simulator_params, self.electrode_grid, self.use_seed)
        self.simulator = PhospheneSimulator(self.simulator_params, self.phosphene_coords)
        self.get_learnable_params = MiniConvNet(self.model_params)
        self.init_weights = torch.load("./init_weights.pth")
        self.get_learnable_params.load_state_dict(self.init_weights, strict=False)
        self.extract_saliency_map = SaliencyMap(self.model_params, requires_grad=True)

    def forward(self, input_image):
        clip_attention_map, saliency_map = self.extract_saliency_map(input_image)
        phosphene_placement_map = self.get_learnable_params(saliency_map.unsqueeze(0).unsqueeze(0))
        # Rescale pixel intensities between [0, <max_stimulation_intensity_ampere>]
        phosphene_placement_map = self.normalized_rescaling(phosphene_placement_map, max_stimulation_intensity=self.simulator_params['sampling']['stimulus_scale'])
        # Make the phosphene_placement_map as a stimulation vector for the phosphene simulator
        phosphene_placement_map = self.simulator.sample_stimulus(phosphene_placement_map)
        self.simulator.reset()
        optimized_im = self.simulator(phosphene_placement_map)
        optimized_im = optimized_im.unsqueeze(0)
        optimized_im = optimized_im.repeat(1, 3, 1, 1)
        optimized_im = optimized_im.permute(0, 1, 2, 3)
        del clip_attention_map, saliency_map
        return optimized_im

    def normalized_rescaling(self, phosphene_placement_map, max_stimulation_intensity=1):  # 100e-6
        """Normalize <img> and rescale the pixel intensities in the range [0, <stimulus_scale>].
        <stimulus_scale> is defined in the parameter file.
        The output image represents the stimulation intensity map.
        return: image with rescaled pixel values (stimulation intensity map in Ampères)."""

        img_norm = (phosphene_placement_map - phosphene_placement_map.min()) / (phosphene_placement_map.max() - phosphene_placement_map.min())
        return img_norm * max_stimulation_intensity

