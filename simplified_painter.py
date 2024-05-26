import os
import PIL
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_pil_image

import CLIP_.clip as clip
import config
import dynaphos
import sketch_utils as sketch_utils
from dynaphos import utils, cortex_models
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator

args = config.parse_arguments()

# -------------------------
# Define model classes
# -------------------------
class Phosphene_model(nn.Module):
    def __init__(self, args,
                 electrode_grid=None,
                 imsize=224,
                 device=None):
        super(Phosphene_model, self).__init__()

        abs_path = os.path.abspath(os.getcwd())
        self.args = args
        self.params = utils.load_params(f"{abs_path}/dynaphos/config/params.yaml")
        self.device = device

        self.canvas_width, self.canvas_height = imsize, imsize
        self.saliency_clip_model = args.saliency_clip_model
        self.text_target = args.text_target

        self.electrode_grid = electrode_grid
        self.num_phosphenes_control = args.num_phosphenes_control
        self.phosphene_coords = cortex_models.get_visual_field_coordinates_probabilistically(self.params,
                                                                                             self.electrode_grid)
        self.stn = PhospheneTransformerNet(size=self.canvas_width, args=self.args)
        self.dynaphos = dynaphos
        self.control_condition = args.control_condition
        self.simulator = PhospheneSimulator(self.params, self.phosphene_coords, self.num_phosphenes_control, self.control_condition)

        self.cached_attention_map = None
        self.cached_clip_saliency_map = None

    def get_clip_saliency_map(self, args, target_im):
        clip_model, preprocess = clip.load(self.saliency_clip_model, device=self.device,
                                           jit=False)
        clip_model.eval().to(self.device)

        data_transforms = transforms.Compose([
            preprocess.transforms[-1],
        ])
        target_im[target_im == 0.] = 1.
        image_input_clip = data_transforms(target_im).to(args.device)
        text_input_clip = clip.tokenize([self.text_target]).to(args.device)

        attention_map = interpret(image_input_clip, text_input_clip, clip_model,
                                  device=args.device)

        del clip_model  # Essential, otherwise CLIP is included in the model parameters

        attn_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

        xdog = XDoG_()
        im_xdog = xdog(image_input_clip[0].permute(1, 2, 0).cpu().numpy(), k=10)
        intersec_map = (1 - im_xdog) * attn_map  # Multiplication of attention map and edge map

        clip_saliency_map = np.copy(intersec_map)
        clip_saliency_map[intersec_map > 0] = softmax(intersec_map[intersec_map > 0],
                                                      tau=args.softmax_temp)
        clip_saliency_map = torch.Tensor(clip_saliency_map) / clip_saliency_map.max()
        clip_saliency_map.requires_grad = True

        return attention_map, clip_saliency_map

    def forward(self, target_im, args, refresh_maps=False):
        if self.cached_clip_saliency_map is None or refresh_maps:
            self.cached_attention_map, self.cached_clip_saliency_map = self.get_clip_saliency_map(args, target_im)
            # Use self.cached_clip_saliency_map in the forward process

        phosphene_placement_map = self.stn(
            self.cached_clip_saliency_map.unsqueeze(0).unsqueeze(0))

        phosphene_placement_map = normalized_rescaling(phosphene_placement_map)
        phosphene_placement_map = self.simulator.sample_stimulus(phosphene_placement_map,
                                                                 rescale=False) #Comment out for random initialization

        self.simulator.reset()
        optimized_im = self.simulator(phosphene_placement_map)
        optimized_im = optimized_im.unsqueeze(0)
        optimized_im = optimized_im.repeat(1, 3, 1, 1)
        optimized_im = optimized_im.permute(0, 1, 2, 3)

        return optimized_im

    def save_png(self, output_dir, name, optimized_im):
        canvas_size = (self.canvas_width, self.canvas_height)

        to_pil = ToPILImage()
        img_pil = to_pil(optimized_im.squeeze(0).squeeze(0))

        img_pil.save('{}/{}.png'.format(output_dir, name), format='PNG', size=canvas_size)


# STN for initialization with saliency map
class PhospheneTransformerNet(nn.Module):
    def __init__(self, size, args):
        super(PhospheneTransformerNet, self).__init__()
        self.size = size
        self.electrode_grid = args.electrode_grid

        self.localization = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, stride=1, padding=0),
            nn.ReLU()
        )

        self.conv_padding = nn.Sequential(
            nn.Upsample((224, 224), mode='nearest'))

    def forward(self, x):
        xs = self.localization(x)
        theta = self.conv_padding(xs)
        theta = torch.clamp(theta, min=theta.mean(), max=theta.max())
        theta = torch.sigmoid(theta)

        return theta


# Original where you get a 1D stimulation tensor, this is random initialization and you dont use the sample_stimulus
# function of the dynaphos
# class PhospheneTransformerNet(nn.Module):
#     def __init__(self, size, args):
#         super(PhospheneTransformerNet, self).__init__()
#         self.size = size
#         self.electrode_grid = args.electrode_grid
#         # Using Sequential for compactness
#         self.localization = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(8, 10, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True)
#         )
#         # Fully connected layer
#         self.fc_loc = nn.Sequential(
#             nn.Linear(10 * 52 * 52, self.electrode_grid))
#
#         # Initialize weights manually
#         self._init_weights()
#
#     def _init_weights(self):
#         # Manually initialize weights for convolutional layers within Sequential
#         for layer in self.localization:
#             if isinstance(layer, nn.Conv2d):
#                 init.kaiming_uniform_(layer.weight, mode='fan_out', nonlinearity='relu')
#                 if layer.bias is not None:
#                     init.constant_(layer.bias, 0)
#
#         # Manually initialize weights for the fully connected layer
#         for layer in self.fc_loc:
#             if isinstance(layer, nn.Linear):
#                 init.xavier_normal_(layer.weight)
#                 init.constant_(layer.bias, 0)
#
#     def forward(self, x):
#         xs = self.localization(x)
#         xs = torch.flatten(xs, start_dim=1)
#
#         theta = self.fc_loc(xs)
#
#         return theta

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


def interpret(image, texts, model, device):
    images = image.repeat(1, 1, 1, 1)
    res = model.encode_image(images)
    model.zero_grad()
    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)
    cams = []  # there are 12 attention blocks
    for i, blk in enumerate(image_attn_blocks):
        cam = blk.attn_probs.detach()
        # each patch is 7x7, so we have 49 pixels + 1 for positional encoding
        cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0)
        cam = cam.clamp(min=0).mean(dim=1)
        cams.append(cam)
        R = R + torch.bmm(cam, R)

    cams_avg = torch.cat(cams)
    cams_avg = cams_avg[:, 0, 1:]
    image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bicubic')
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    return image_relevance


def softmax(x, tau=0.2):
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()

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


class XDoG_(object):
    def __init__(self):
        super(XDoG_, self).__init__()
        self.gamma = 0.98
        self.phi = 200
        self.eps = -0.1
        self.sigma = 0.8
        self.binarize = True

    def __call__(self, im, k=10):
        if im.shape[2] == 3:
            im = rgb2gray(im)
        imf1 = gaussian_filter(im, self.sigma)
        imf2 = gaussian_filter(im, self.sigma * k)
        imdiff = imf1 - self.gamma * imf2
        imdiff = (imdiff < self.eps) * 1.0 + (imdiff >= self.eps) * (1.0 + np.tanh(self.phi * imdiff))
        imdiff -= imdiff.min()
        imdiff /= imdiff.max()
        if self.binarize:
            th = threshold_otsu(imdiff)
            imdiff = imdiff >= th
        imdiff = imdiff.astype('float32')
        return imdiff
