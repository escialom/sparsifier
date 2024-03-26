import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt

import dynaphos
import CLIP_.clip as clip
from PIL import Image
from dynaphos import utils
from dynaphos import cortex_models
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator
from config import args
import sketch_utils as utils
import PIL
from scipy.ndimage.filters import gaussian_filter
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from torchvision import transforms


# preprocessing: Getting target, preprocessing and getting saliency map

def get_target_and_mask(args):
    target = Image.open(args.target)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    masked_im, mask = utils.get_mask_u2net(args, target)
    if args.mask_object:
        target = masked_im
    if args.fix_scale:
        target = utils.fix_image_scale(target)

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
        cam = blk.attn_probs.detach()  # attn_probs shape is 12, 50, 50
        # each patch is 7x7, so we have 49 pixels + 1 for positional encoding
        cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0)
        cam = cam.clamp(min=0).mean(dim=1)  # mean of the 12 something
        cams.append(cam)
        R = R + torch.bmm(cam, R)

    cams_avg = torch.cat(cams)  # 12, 50, 50
    cams_avg = cams_avg[:, 0, 1:]  # 12, 1, 49
    image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bicubic')
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    return image_relevance


def softmax(self, x, tau=0.2):
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()


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


target_im, mask = get_target_and_mask(args)

data_transforms = transforms.Compose([
    preprocess.transforms[-1], ])

image_input_clip = data_transforms(target_im).to(args.device)
text_input_clip = clip.tokenize([args.text_target]).to(args.device)

attention_map = interpret(image_input_clip, text_input_clip, clip_model, device=args.device)
del clip_model

attn_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

xdog = XDoG_()
im_xdog = xdog(image_input_clip[0].permute(1, 2, 0).cpu().detach().numpy(), k=10)
intersec_map = (1 - im_xdog) * attn_map

attn_map_soft = np.copy(intersec_map)
attn_map_soft[intersec_map > 0] = softmax(intersec_map[intersec_map > 0], tau=args.softmax_temp)

# And do preprocessing for the CLIP model


class Painter_model(nn.Module):
    def __init__(self, args,
                 num_phosphenes=args.num_phosphenes,
                 imsize=224,
                 device=None):
        super(Painter_model, self).__init__()

        self.args = args
        self.num_phosphenes = num_phosphenes  # Number of electrodes or phosphenes to initialize the electrode grid with
        self.constrain = args.constrain  # Related to if we want to have number of phosphenes constrained
        self.percentage = args.percentage  # Indicates the percentage of phosphenes to keep
        self.device = device
        self.canvas = args.canvas  # Canvas
        self.canvas_width, self.canvas_height = imsize, imsize  # Canvas size
        self.params = utils.load_params(
            'C:/Users/vanholk/sparsifier/dynaphos/config/params.yaml')  # TODO move this to our config file
        self.params['run']['fps'] = 10  # 10 fps -> a single frame represents 100 milliseconds

        self.stn = PhospheneTransformerNet(size=self.canvas_width)  # Spatial Transformer Network
        self.dynaphos = dynaphos  # Dynaphos
        self.clip_model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)  # CLIP model
        self.clip_model.eval().to(self.device)

        self.phosphene_coords = cortex_models.get_visual_field_coordinates_probabilistically(self.params,
                                                                                             self.num_phosphenes)
        self.simulator = PhospheneSimulator(self.params, self.phosphene_coords)

    def forward(self, init_sketch, target_im):
        # Put attention map in stn to get inds
        self.inds = self.stn(torch.Tensor(attn_map_soft).unsqueeze(0).unsqueeze(0))
        self.inds_normalised = (self.inds / self.inds.max()).to_list()

        inds_normalised_rescaled = normalized_rescaling(self.inds_normalised)
        stim_inds = self.simulator.sample_stimulus(inds_normalised_rescaled)
        stim_inds = stim_inds.detach()  # Otherwise the loss.backward() does not work

        self.simulator.reset()

        phosphenes = self.simulator(stim_inds)

        if self.constrain:
            original_phosphenes_sum = phosphenes.sum().item()
            max_total_current = original_phosphenes_sum * (self.percentage / 100.0)
            adjusted_phosphenes = randomly_deactivate_phosphenes(phosphenes.clone(), max_total_current)
            phosphenes = adjusted_phosphenes

        plot_init_phosphenes(phosphenes)

    #And do preprocessing for CLIP model

        return phosphenes


class PhospheneTransformerNet(nn.Module):
    def __init__(self, size):
        super(PhospheneTransformerNet, self).__init__()
        self.size = size
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 52 * 52, 1024),  # Adjusted input dimension
            nn.ReLU(True),
            nn.Linear(1024, 224 * 224)  # TODO: change this to a vector/matrix given to dynaphos
        )

    def forward(self, x):
        xs = self.localization(x)
        # Ensure xs is correctly reshaped for the fully connected layers
        # The reshaping depends on the output size of your localization layers
        num_features = 10 * 52 * 52
        xs = xs.view(-1, num_features)  # num_features needs to be calculated based on the STN design
        theta = self.fc_loc(xs)
        theta = theta.view(224, 224)
        theta = (theta - theta.min()) / (theta.max() - theta.min())

        return theta


def normalized_rescaling(img, stimulus_scale=100.e-6):
    """Normalize <img> and rescale the pixel intensities in the range [0, <stimulus_scale>].
    The output image represents the stimulation intensity map.
    param stimulus_scale: the stimulation amplitude corresponding to the highest-valued pixel.
    return: image with rescaled pixel values (stimulation intensity map in Ampères)."""

    img_norm = (img - img.min()) / (img.max() - img.min())
    return img_norm * stimulus_scale


def randomly_deactivate_phosphenes(phosphenes, max_total_current):
    """
    Randomly deactivates a selection of phosphenes to ensure the total current
    does not exceed the specified maximum value. This function turns off phosphenes
    completely without changing the intensity of the remaining ones.

    Parameters:
    phosphenes (Tensor): The original phosphenes activation map.
    max_total_current (float): The maximum allowed sum of intensities over the electrode grid (total current).

    Returns:
    Tensor: The adjusted phosphenes intensity map.
    """
    # Flatten the phosphenes for easier manipulation
    original_shape = phosphenes.shape
    phosphenes_flat = phosphenes.flatten()

    # Keep reducing phosphenes until the total current is under the limit
    while phosphenes_flat.sum() > max_total_current:
        # Find indices of currently active (non-zero) phosphenes
        active_indices = torch.nonzero(phosphenes_flat, as_tuple=True)[0]

        # If no active phosphenes left but still over max_current, break to avoid infinite loop
        if len(active_indices) == 0:
            break

        # Randomly select one active phosphene to turn off
        index_to_turn_off = np.random.choice(active_indices.cpu().numpy(), 1)

        # Turn off the selected phosphene
        phosphenes_flat[index_to_turn_off] = 0

    # Reshape back to the original phosphenes map
    adjusted_phosphenes = phosphenes_flat.reshape(original_shape)

    return adjusted_phosphenes

def plot_init_phosphenes(phosphenes):
    img = phosphenes
    # Convert img from HW to NHW
    img = img.unsqueeze(0)
    img = img.repeat(1, 3, 1, 1)  # Now the shape is [1, 3, H, W]
    img = img.permute(0, 1, 2, 3).to(args.device)  # NHW -> NHW
    # Convert tensor to numpy array
    img_np = img.squeeze().cpu().detach().numpy()
    # Transpose the dimensions from [C, H, W] to [H, W, C] for RGB image
    img_np = img_np.transpose(1, 2, 0)

    # Plot the image
    plt.imshow((img_np * 255).astype(int), cmap='gray')
    plt.axis('off')
    plt.title('Initialized image')
    plt.show()

    return img
