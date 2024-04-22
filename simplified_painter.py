import torch.nn as nn
import numpy as np
import torch
import torch.nn.init as init
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

# Importing necessary libraries and modules
import dynaphos
import CLIP_.clip as clip
from PIL import Image
from dynaphos import utils, cortex_models
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator
import config
import sketch_utils as utils
import PIL
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from torchvision import transforms
from PIL import Image


# -------------------------
# Define model classes
# -------------------------


class Phosphene_model(nn.Module):
    def __init__(self, args,
                 num_phosphenes=None,
                 imsize=224,
                 device=None):
        super(Phosphene_model, self).__init__()

        self.args = args
        self.num_phosphenes = num_phosphenes  # Number of electrodes or phosphenes to initialize the electrode grid with
        self.constrain = args.constrain  # Related to if we want to have number of phosphenes constrained
        self.percentage = args.percentage  # Indicates the percentage of phosphenes to keep
        self.device = device
        self.canvas_width, self.canvas_height = imsize, imsize  # Canvas size
        self.params = utils.load_params(
            'C:/Users/vanholk/sparsifier/dynaphos/config/params.yaml')  # TODO move this to our config file
        self.params['run']['fps'] = 10  # 10 fps -> a single frame represents 100 milliseconds
        self.saliency_clip_model = args.saliency_clip_model
        self.text_target = args.text_target

        self.stn = PhospheneTransformerNet(size=self.canvas_width, args=self.args)  # Spatial Transformer Network
        self.dynaphos = dynaphos  # Dynaphos

        self.phosphene_coords = cortex_models.get_visual_field_coordinates_probabilistically(self.params,
                                                                                             self.num_phosphenes)
        self.simulator = PhospheneSimulator(self.params, self.phosphene_coords)

        self.cached_attention_map = None
        self.cached_clip_saliency_map = None

    def get_clip_saliency_map(self, args, target_im):
        clip_model, preprocess = clip.load(self.saliency_clip_model, device=self.device,
                                           jit=False)  # CLIP model
        clip_model.eval().to(self.device)

        data_transforms = transforms.Compose([
            preprocess.transforms[-1],
        ])

        image_input_clip = data_transforms(target_im).to(args.device)  # Defining image input for CLIP
        text_input_clip = clip.tokenize([self.text_target]).to(args.device)  # Defining text input for CLIP

        attention_map = interpret(image_input_clip, text_input_clip, clip_model,
                                  device=args.device)  # Make
        # attention map using the interpret method

        del clip_model  # Essential, otherwise CLIP is included in the model parameters

        attn_map = (attention_map - attention_map.min()) / (
                attention_map.max() - attention_map.min())  # Normalization of
        # attention map

        xdog = XDoG_()  # Make instance of XDoG


        # Then apply XDoG and combine
        im_xdog = xdog(image_input_clip[0].permute(1, 2, 0).cpu().numpy(), k=10)
        intersec_map = (1-im_xdog) * attn_map  # Multiplication of attention map and edge map
        # intersec_map = im_xdog * attn_map

        clip_saliency_map = np.copy(intersec_map)
        clip_saliency_map[intersec_map > 0] = softmax(intersec_map[intersec_map > 0],
                                                      tau=args.softmax_temp) #see if this is necessary too

        # PLOT TO SEE WHAT SALIENCY MAP LOOKS LIKE TO SEE WHERE OUR SALIENT REGIONS ARE

        # may have to turn it into a tensor too.
        clip_saliency_map = torch.Tensor(clip_saliency_map) / clip_saliency_map.max()

        # clip_saliency_map = torch.Tensor(clip_saliency_map)
        clip_saliency_map.requires_grad = True

        return attention_map, clip_saliency_map

    def forward(self, target_im, args, refresh_maps=False):

        if self.cached_clip_saliency_map is None or refresh_maps:
            self.cached_attention_map, self.cached_clip_saliency_map = self.get_clip_saliency_map(args, target_im)

            # Use self.cached_clip_saliency_map in the forward process
        phosphene_placement_map = self.stn(self.cached_clip_saliency_map.unsqueeze(0).unsqueeze(0))

        # attention_map, clip_saliency_map = self.get_clip_saliency_map(args, target_im)  # These are good
        # # clip_saliency_map = torch.Tensor(clip_saliency_map) / clip_saliency_map.max()
        # # # clip_saliency_map = torch.Tensor(clip_saliency_map)
        # # clip_saliency_map.requires_grad = True
        #
        # # clip_saliency_map = torch.rand(size=(224, 224), requires_grad=True)
        #
        # # Put attention map in stn to get phosphene placement map
        # phosphene_placement_map = self.stn(clip_saliency_map.unsqueeze(0).unsqueeze(0))
        # phosphene_placement_map = (phosphene_placement_map / phosphene_placement_map.max())

        phosphene_placement_map = normalized_rescaling(phosphene_placement_map)
        # phosphene_placement_map = self.simulator.sample_stimulus(phosphene_placement_map)

        self.simulator.reset()

        phosphene_im = self.simulator(phosphene_placement_map)

        # Find the maximum value across the phosphene image.
        # max_value = phosphene_im.max()
        #
        # # Ensure that all non-zero phosphenes have the same intensity, based on the brightest one.
        # # This operation preserves the shape of phosphenes but sets all non-zero intensities to the maximum found.
        # phosphene_im = torch.where(phosphene_im > phosphene_im.min(), max_value, phosphene_im) # TODO this gives a very rough image
        # but it is just for a proof of principle

        if self.constrain:
            original_phosphenes_sum = phosphene_im.sum().item()
            max_total_current = original_phosphenes_sum * (self.percentage / 100.0)
            adjusted_phosphenes = randomly_deactivate_phosphenes(phosphene_im.clone(), max_total_current)
            phosphene_im = adjusted_phosphenes

        # plot_init_phosphenes(phosphenes)

        # min_value = 0.1  # Set the minimum value to make the pixels visible

        # Apply conditional operation to set a minimum value for non-zero pixels in phosphene_im
        # phosphene_im = torch.where(phosphene_im > 0, torch.maximum(phosphene_im, torch.tensor(min_value)), phosphene_im)

        phosphene_im = phosphene_im.unsqueeze(0)
        phosphene_im = phosphene_im.repeat(1, 3, 1, 1)  # Now the shape is [1, 3, H, W]
        phosphene_im = phosphene_im.permute(0, 1, 2, 3)  # .to(self.device)  # NHW -> NHW

        return phosphene_im

    def save_png(self, output_dir, name, phosphene_im):
        canvas_size = (self.canvas_width, self.canvas_height)

        to_pil = ToPILImage()
        img_pil = to_pil(phosphene_im.squeeze(0).squeeze(0))

        img_pil.save('{}/{}.png'.format(output_dir, name), format='PNG', size=canvas_size)


class PhospheneTransformerNet(nn.Module):
    def __init__(self, size, args):
        super(PhospheneTransformerNet, self).__init__()
        self.size = size
        self.num_phosphenes = args.num_phosphenes
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding=3),
            nn.ELU(True),  # Softer non-linearity
            nn.AvgPool2d(2, stride=2),  # Gentle pooling
            nn.Conv2d(8, 10, kernel_size=5, padding=2),
            nn.ELU(True),  # Softer non-linearity
            nn.AvgPool2d(2, stride=2)  # Gentle pooling
        )
        # Assuming size is the dimension of the input image, adjust based on actual input size
        # Pooling twice with stride 2 reduces each dimension by a factor of 4
        reduced_dim = size // 4
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * reduced_dim * reduced_dim, self.num_phosphenes)
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize with identity-like structures where possible
        for layer in self.localization:
            if isinstance(layer, nn.Conv2d):
                init.constant_(layer.weight, 0)  # Set all weights to zero
                center = tuple(map(lambda x: x // 2, layer.kernel_size))
                for i in range(layer.out_channels):
                    for j in range(layer.in_channels):
                        layer.weight.data[i, j, center[0], center[1]] = 1  # Set center weight to 1

                if layer.bias is not None:
                    init.constant_(layer.bias, 0)

        # Fully connected layers initialized more traditionally but consider identity where possible
        for layer in self.fc_loc:
            if isinstance(layer, nn.Linear):
                identity_size = min(layer.weight.shape[0], layer.weight.shape[1])
                init.constant_(layer.weight, 0)
                with torch.no_grad():
                    for i in range(identity_size):
                        layer.weight[i, i] = 1
                init.constant_(layer.bias, 0)

    def forward(self, x):
        xs = self.localization(x)
        xs = torch.flatten(xs, start_dim=1)
        theta = self.fc_loc(xs)
        return theta


# class PhospheneTransformerNet(nn.Module):
#     def __init__(self, size, args):
#         super(PhospheneTransformerNet, self).__init__()
#         self.size = size
#         self.num_phosphenes = args.num_phosphenes
#         # Using Sequential for compactness
#         self.localization = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=7),
#             nn.MaxPool2d(2, stride=2), #stride =2
#             nn.ReLU(True),
#             nn.Conv2d(8, 10, kernel_size=5),
#             nn.MaxPool2d(2, stride=2), #stride =2
#             nn.ReLU(True)
#         )
#         # Fully connected layer
#         self.fc_loc = nn.Sequential(
#             nn.Linear(10 * 52 * 52, self.num_phosphenes))  # change last number to number of phosphenes
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
# #
#     def forward(self, x):
#         xs = self.localization(x)
#         xs = torch.flatten(xs, start_dim=1)
#
#         theta = self.fc_loc(xs)
#
#         return theta


# def forward(self, x):
#     xs = self.localization(x)
#     # Ensure xs is correctly reshaped for the fully connected layers
#     # The reshaping depends on the output size of your localization layers
#     num_features = 10 * 52 * 52
#     xs = xs.view(-1, num_features)  # num_features needs to be calculated based on the STN design
#     theta = self.fc_loc(xs)
#     theta = F.softmax(theta)
#     #theta = theta.view(224, 224)
#     theta = theta.view(32, 32)
#     # theta = (theta - theta.min()) / (theta.max() - theta.min())

# return theta


# -------------------------
# Utility functions
# -------------------------

def normalized_rescaling(img, stimulus_scale=1000e-6):  # 100e-6
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


def plot_init_phosphenes(args, phosphenes):
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


def get_target_and_mask(args, target_image_path):
    target = Image.open(target_image_path)
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
    image_attn_blocks = list(dict(
        model.visual.transformer.resblocks.named_children()).values())
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
    image_relevance = image_relevance.to(torch.float32)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bicubic')
    # image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    return image_relevance


def softmax(x, tau=0.2):
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()


class XDoG_(object):
    def __init__(self):
        super(XDoG_, self).__init__()
        self.gamma = 0.98 #0.98
        self.phi = 200 #200
        self.eps = -0.1
        self.sigma = 0.8 #0.8
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
