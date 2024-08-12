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

        # self.clip_model, self.preprocess = clip.load(self.model_params.saliency_clip_model, device=self.model_params.device,
        #                                              jit=False)
        # self.clip_model.eval().to(self.model_params.device)

    def forward(self, input_image): #clip_attention_map, saliency_map_soft --> attention_map, saliency_map
        clip_attention_map, saliency_map = self.extract_saliency_map(input_image)
        phosphene_placement_map = self.get_learnable_params(saliency_map.unsqueeze(0).unsqueeze(0))
        # Make the phosphene_placement_map as a stimulation vector for the phosphene simulator
        phosphene_placement_map = self.normalized_rescaling(phosphene_placement_map)
        phosphene_placement_map = self.simulator.sample_stimulus(phosphene_placement_map, rescale=False)
        self.simulator.reset()
        optimized_im = self.simulator(phosphene_placement_map)
        optimized_im = optimized_im.unsqueeze(0)
        optimized_im = optimized_im.repeat(1, 3, 1, 1)
        optimized_im = optimized_im.permute(0, 1, 2, 3)
        del clip_attention_map, saliency_map
        return optimized_im

    def normalized_rescaling(self, phosphene_placement_map):  # 100e-6
        """Normalize <img> and rescale the pixel intensities in the range [0, <stimulus_scale>].
        The output image represents the stimulation intensity map.
        param stimulus_scale: the stimulation amplitude corresponding to the highest-valued pixel.
        return: image with rescaled pixel values (stimulation intensity map in Ampères)."""

        img_norm = (phosphene_placement_map - phosphene_placement_map.min()) / (phosphene_placement_map.max() - phosphene_placement_map.min())
        return img_norm * self.simulator_params['sampling']['stimulus_scale']


#     def get_clip_saliency_map(self, args, target_im):
#
#         data_transforms = transforms.Compose([
#             self.preprocess.transforms[-1],
#         ])
#         target_im[target_im == 0.] = 1.
#         image_input_clip = data_transforms(target_im).to(args.device)
#         text_input_clip = clip.tokenize([self.model_params.text_target]).to(args.device)
#
#         attention_map = interpret(image_input_clip, text_input_clip, self.clip_model,
#                                   device=args.device)
#
#         attn_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
#
#         xdog = clipasso_model.XDoG_()
#         im_xdog = xdog(image_input_clip[0].permute(1, 2, 0).cpu().numpy(), k=10)
#         intersec_map = (1 - im_xdog) * attn_map  # Multiplication of attention map and edge map
#
#         clip_saliency_map = np.copy(intersec_map)
#         clip_saliency_map[intersec_map > 0] = self.softmax(intersec_map[intersec_map > 0],
#                                                       tau=args.softmax_temp)
#         clip_saliency_map = torch.Tensor(clip_saliency_map) / clip_saliency_map.max()
#         clip_saliency_map.requires_grad = True
#
#         return attention_map, clip_saliency_map
#
#     def softmax(self, x, tau=0.2):
#         e_x = np.exp(x / tau)
#         return e_x / e_x.sum()
#
# def interpret(image, texts, model, device):
#     images = image.repeat(1, 1, 1, 1)
#     res = model.encode_image(images)
#     model.zero_grad()
#     image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
#     num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
#     R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
#     R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)
#     cams = []  # there are 12 attention blocks
#     for i, blk in enumerate(image_attn_blocks):
#         cam = blk.attn_probs.detach()
#         # each patch is 7x7, so we have 49 pixels + 1 for positional encoding
#         cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
#         cam = cam.clamp(min=0)
#         cam = cam.clamp(min=0).mean(dim=1)
#         cams.append(cam)
#         R = R + torch.bmm(cam, R)
#
#     cams_avg = torch.cat(cams)
#     cams_avg = cams_avg[:, 0, 1:]
#     image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
#     image_relevance = image_relevance.reshape(1, 1, 7, 7)
#     image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bicubic')
#     image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
#     image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
#     return image_relevance
#
#
#
