import os
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_pil_image
import clipasso.CLIP_.clip as clip
import config
import dynaphos
import sketch_utils as sketch_utils
from dynaphos import utils, cortex_models
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator

from clipasso.models import painter_params as clipasso_model #
args = config.model_config.parse_arguments()

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
        saliency_map_soft[saliency_map > 0] = self.softmax(saliency_map[saliency_map > 0], tau=args.softmax_temp)
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
    def __init__(self, args,
                 electrode_grid=None,
                 imsize=224,
                 device=None):
        super(PhospheneOptimizer, self).__init__()

        self.get_learnable_params = MiniConvNet()


        abs_path = os.path.abspath(os.getcwd())
        self.args = args
        self.params = utils.load_params(f"{abs_path}/dynaphos/config/params.yaml")
        self.device = device

        self.canvas_width, self.canvas_height = imsize, imsize
        self.saliency_clip_model = args.saliency_clip_model
        self.text_target = args.text_target

        self.electrode_grid = electrode_grid
        self.num_phosphenes_control = args.num_phosphenes_control
        self.phosphene_selection = args.phosphene_selection
        self.phosphene_density = args.phosphene_density
        self.phosphene_coords = cortex_models.get_visual_field_coordinates_probabilistically(self.params,
                                                                                             self.electrode_grid)
        self.dynaphos = dynaphos
        self.control_condition = args.control_condition
        self.simulator = PhospheneSimulator(self.params, self.phosphene_coords, self.num_phosphenes_control,
                                            self.phosphene_selection, self.phosphene_density,
                                            control_condition=self.control_condition)
        self.clip_model, self.preprocess = clip.load(self.saliency_clip_model, device=self.device,
                                                     jit=False)
        self.clip_model.eval().to(self.device)

    def forward(self, input_image, args): #clip_attention_map, saliency_map_soft --> attention_map, saliency_map
        clip_attention_map, saliency_map = SaliencyMap(input_image)
        phosphene_placement_map = self.get_learnable_params(saliency_map.unsqueeze(0).unsqueeze(0))
        phosphene_placement_map = normalized_rescaling(phosphene_placement_map) # check dynaphos
        phosphene_placement_map = self.simulator.sample_stimulus(phosphene_placement_map,
                                                                 rescale=False)  # Comment out for random initialization

        self.simulator.reset()
        optimized_im = self.simulator(phosphene_placement_map)
        # Save the number of phosphenes per iteration to a CSV file
        save_path = os.path.join(args.output_dir, 'phosphenes_count.csv')
        self.simulator.save_phosphenes_count(save_path)
        optimized_im = optimized_im.unsqueeze(0)
        optimized_im = optimized_im.repeat(1, 3, 1, 1)
        optimized_im = optimized_im.permute(0, 1, 2, 3)

        del clip_attention_map, saliency_map

        return optimized_im





# -------------------------
# Utility functions
# -------------------------
def normalized_rescaling(img, stimulus_scale=args.stimulus_scale_optimized):  # 100e-6
    """Normalize <img> and rescale the pixel intensities in the range [0, <stimulus_scale>].
    The output image represents the stimulation intensity map.
    param stimulus_scale: the stimulation amplitude corresponding to the highest-valued pixel.
    return: image with rescaled pixel values (stimulation intensity map in Ampères)."""

    img_norm = (img - img.min()) / (img.max() - img.min())
    return img_norm * stimulus_scale


def get_target_and_mask(args, target_image_path):
    target = Image.open(target_image_path)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    masked_im, mask = sketch_utils.get_mask_u2net(args, target)
    if args.mask_object:
        target = masked_im
    if args.fix_scale:
        target = sketch_utils.fix_image_scale(target)

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






def plot_saliency_map(target_im, attention_map, clip_saliency_map):
    """
    Plots the target image, attention map, and CLIP saliency map side by side.

    :param target_im: Target image tensor [C, H, W].
    :param attention_map: Attention map as a numpy array.
    :param clip_saliency_map: CLIP saliency map as a numpy array.
    """
    # Convert the target image tensor to PIL for consistent plotting
    target_pil = to_pil_image(target_im.squeeze(0))

    # Set up the matplotlib figure and axes
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot each of the images/maps
    axs[0].imshow(target_pil)
    axs[0].set_title('Target Image')
    axs[0].axis('off')

    axs[1].imshow(attention_map, cmap='viridis')
    axs[1].set_title('Attention Map')
    axs[1].axis('off')

    axs[2].imshow(clip_saliency_map.detach().numpy(), cmap='viridis')
    axs[2].set_title('CLIP Saliency Map')
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/saliency_map.png", bbox_inches='tight')
    plt.close()

