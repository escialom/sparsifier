import random
import CLIP_.clip as clip
import numpy as np
import sketch_utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from torchvision import transforms
from torch import Tensor
from torchvision.transforms import ToPILImage
import dynaphos
from dynaphos import utils
from dynaphos import cortex_models
from dynaphos.cortex_models import get_visual_field_coordinates_from_cortex_full, Map
from dynaphos.image_processing import canny_processor, sobel_processor
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator
from dynaphos.utils import get_data_kwargs, to_numpy

import cv2


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
            nn.Linear(10 * 52 * 52, 128),  # Adjusted input dimension
            nn.ReLU(True),
            nn.Linear(128, 16 * 2)  # TODO: change this to a vector/matrix given to dynaphos
        )

        # Initialize the weights/bias with identity transformation
        #self.fc_loc[2].weight.data.zero_()
        #self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        # Ensure xs is correctly reshaped for the fully connected layers
        # The reshaping depends on the output size of your localization layers
        num_features = 10 * 52 * 52
        xs = xs.view(-1, num_features)  # num_features needs to be calculated based on the STN design
        theta = self.fc_loc(xs)
        theta = theta.view(16, 2)
        theta = (theta - theta.min()) / (theta.max() - theta.min())

        ## Generate the grid
        #grid = F.affine_grid(theta, x.size(), align_corners=False)
        return theta



class Painter(torch.nn.Module):
    def __init__(self, args,
                 num_phosphenes = 4,
                 #num_strokes=4,
                 # num_segments=4,
                 imsize=224,
                 device=None,
                 target_im=None,
                 mask=None):
        super(Painter, self).__init__()

        self.args = args
        self.num_phosphenes = num_phosphenes
        # self.num_segments = num_segments
        # self.width = args.width
        # self.control_points_per_seg = args.control_points_per_seg
        # self.opacity_optim = args.force_sparse
        self.num_stages = args.num_stages
        self.add_random_noise = "noise" in args.augmentations
        self.noise_thresh = args.noise_thresh
        self.softmax_temp = args.softmax_temp

        self.point_locations = []
        # self.shape_groups = []
        self.device = device
        self.canvas = args.canvas
        self.canvas_width, self.canvas_height = imsize, imsize
        self.patch_size = args.patch_size
        self.points_vars = []
        # self.color_vars = []
        # self.color_vars_threshold = args.color_vars_threshold

        self.path_svg = args.path_svg
        self.points_per_stage = self.num_phosphenes
        self.optimize_flag = []
        self.output_dir = args.output_dir

        # attention related for strokes initialisation
        self.attention_init = args.attention_init
        self.target_path = args.target
        self.saliency_model = args.saliency_model
        self.xdog_intersec = args.xdog_intersec
        self.mask_object = args.mask_object_attention
        self.stn = PhospheneTransformerNet(size=self.canvas_width)

        self.text_target = args.text_target # for clip gradients
        self.saliency_clip_model = args.saliency_clip_model
        self.define_attention_input(target_im)
        self.mask = mask
        self.attention_map = self.set_attention_map() if self.attention_init else None

        self.thresh = self.set_attention_threshold_map() if self.attention_init else None
        self.points_counter = 0 # counts the number of calls to "get_path"
        self.epoch = 0
        self.final_epoch = args.num_iter - 1

    def rescale_positions(self, hard_selection):
        output_size = [1, 1, self.canvas_width, self.canvas_height]

        # Create the upscaling grid
        grid = self.create_upscaling_grid(output_size)

        # Use grid_sample for differentiable upscaling
        upscaled_selection = F.grid_sample(hard_selection.unsqueeze(0).unsqueeze(0), grid, mode='nearest',
                                           align_corners=False)

        return upscaled_selection.squeeze(0)

    def create_upscaling_grid(self, output_size):
        # Calculate scale factors for rows and columns
        # scale_factor = hard_selection.shape[0] / output_size[1] # Assuming square input and output
        scale_factor = 1.

        # Create an affine transformation that scales up the image
        # The translation components are set to 0 for no translation
        theta = torch.tensor([[scale_factor, 0, 0], [0, scale_factor, 0]], dtype=torch.float).unsqueeze(0)

        # Generate a grid for the transformation
        grid = F.affine_grid(theta, size=output_size, align_corners=False)

        return grid

    def init_image(self, stage=0):
        if stage > 0:
            # if multi stages training than add new strokes on existing ones
            # don't optimize on previous strokes
            self.optimize_flag = [False for i in range(len(self.point_locations))]
            for i in range(self.points_per_stage):
                # point_color = torch.tensor([0.0, 0.0, 0.0, 1.0]) #was stroke_color
                point_locations = self.get_point_locations()
                self.point_locations.append(point_locations)

                # path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                #                                     fill_color = None,
                #                                     stroke_color = stroke_color)
                # self.shape_groups.append(path_group)
                self.optimize_flag.append(True)

        else:
            num_phosphenes_exists = 0
            if self.path_svg != "none":
                self.canvas_width, self.canvas_height, self.shapes, self.shape_groups = utils.load_svg(self.path_svg)
                # if you want to add more strokes to existing ones and optimize on all of them
                num_phosphenes_exists = len(self.point_locations)

                # stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            point_locations = self.get_point_locations().squeeze(0)
            self.point_locations.append(point_locations)
            self.point_locations = self.point_locations[0]
            print(self.point_locations)

            # path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
            #                                     fill_color = None,
            #                                     stroke_color = stroke_color)
            # self.shape_groups.append(path_group)
            self.optimize_flag = [True for i in range(len(self.point_locations))]

        img = self.render_warp()
        # img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = self.device) * (1 - img[:, :, 3:4])
        # img = img[:, :, :3]
        # Convert img from HW to NHW
        img = img.unsqueeze(0)
        img = img.repeat(1, 3, 1, 1)  # Now the shape is [1, 3, H, W]
        img = img.permute(0, 1, 2, 3).to(self.device)  # NHW -> NHW
        # Convert tensor to numpy array
        img_np = img.squeeze().cpu().detach().numpy()
        # Transpose the dimensions from [C, H, W] to [H, W, C] for RGB image
        img_np = img_np.transpose(1, 2, 0)

        # Plot the image
        imshow((img_np*255).astype(int), cmap='gray')
        plt.axis('off')
        plt.title('Initialized image')
        plt.show()

        # Or save the image
        # img_pil = Image.fromarray((img_np * 255).astype('uint8'))  # Convert to PIL Image
        # img_pil.save( f"{self.output_dir}/camel_initialized_image.png")  # Save the image
        return img
        # utils.imwrite(img.cpu(), '{}/init.png'.format(args.output_dir), gamma=args.gamma, use_wandb=args.use_wandb, wandb_name="init")

    def get_image(self, mock=None):
        img = self.render_warp()
        img = img.unsqueeze(0)
        img = img.repeat(1, 3, 1, 1)  # Now the shape is [1, 3, H, W]
        img = img.permute(0, 1, 2, 3).to(self.device)  # NHW -> NHW
        return img

    def get_point_locations(self) -> Tensor:
        point_locations = []

        #self.num_control_points = torch.zeros(self.num_segments, dtype=torch.int32) + (self.control_points_per_seg - 2)
        phosphene_centers = self.inds_normalised if self.attention_init else (random.random(), random.random())
        point_locations.append(phosphene_centers)

        point_locations = torch.tensor(point_locations).to(self.device)
        point_locations = point_locations.squeeze()

        point_locations[:, 0] *= self.canvas_width
        point_locations[:, 1] *= self.canvas_height

        return point_locations

    def generate_phosphene(self) -> Tensor:
        """Generates a phosphene with a given radius on a patch."""
        # Define grid of phosphene
        half_patch = int(self.patch_size // 2)
        x = torch.arange(start=-half_patch, end=half_patch + 1)
        x_grid, y_grid = torch.meshgrid([x, x])

        # Generate phosphene on the grid. Luminance values normalized between 0 and 1
        phosphene = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * self.args.phosphene_radius ** 2))
        phosphene /= phosphene.max()

        # Min-Max normalization to scale phosphene to [0, 1]
        min_val, max_val = phosphene.min(), phosphene.max()
        phosphene_scaled = (phosphene - min_val) / (max_val - min_val)
        phosphene = phosphene_scaled

        # Gamma Correction with gamma < 1 to brighten the values
        # gamma = 0.5 # Random value
        # phosphene_corrected = phosphene_scaled ** gamma
        # phosphene = phosphene_corrected

        return phosphene.unsqueeze(0)

    @staticmethod
    def calculate_padding(x_pos: int, y_pos: int, smoothed_element: Tensor, canvas: Tensor):
        pad_top = max(y_pos - smoothed_element.shape[2] // 2, 0)
        pad_left = max(x_pos - smoothed_element.shape[1] // 2, 0)
        pad_bottom = max(canvas.shape[1] - pad_top - smoothed_element.shape[2], 0)
        pad_right = max(canvas.shape[0] - pad_left - smoothed_element.shape[1], 0)
        return (pad_left, pad_right, pad_top, pad_bottom)

    def _render(self) -> torch.Tensor:
        # TODO: replace with dynaphos somehow
        # elements = self.generate_phosphene()
        #
        # elem_xy_locations = self.point_locations
        # # elem_xy_locations_pre = self.point_locations
        # # elem_xy_locations: torch.Tensor = elem_xy_locations_pre[0]
        # canvas = self.canvas.clone()
        # output_img = torch.zeros_like(canvas)
        #
        # # Compute the scaling factor
        # # scale_x = canvas.shape[0] / elem_xy_locations.shape[0]
        # # scale_y = canvas.shape[1] / elem_xy_locations.shape[1]
        #
        # for coord in elem_xy_locations:
        #     # Extract the actual x and y positions from each coordinate pair
        #     x_pos, y_pos = coord[0], coord[1]
        #
        #     padding = self.calculate_padding(x_pos, y_pos, elements, canvas)
        #     padded_element = F.pad(elements, padding)
        #
        #     blending_weight = 1 # Some arbitrary value now
        #     # Apply weighted blending
        #     output_img += blending_weight * padded_element[0]
        #
        # # for x in range(elem_xy_locations.shape[0]):
        # #     for y in range(elem_xy_locations.shape[1]):
        # #         if elem_xy_locations[x, y] != 0:
        # #             # Calculate the scaled positions
        # #             # x_pos, y_pos = int(x * scale_x), int(y * scale_y)
        # #             x_pos, y_pos = int(x), int(y)
        # #             # pad the element to the canvas size
        # #             padding = self.calculate_padding(x_pos, y_pos, elements, canvas)
        # #             padded_element = F.pad(elements, padding)
        # #             # blend the element locations to the output
        # #             blending_weight = torch.sigmoid(elem_xy_locations[x, y])
        # #             # Apply weighted blending
        # #             output_img += blending_weight * padded_element[0]
        # #
        # return output_img

        #TODO so here we want to as output an image with phosphenes rendered through dynaphos
        # So instead of elements = our phosphene generator, we want elements = dynaphos

        #
        # Load the simulator configuration file
        params = utils.load_params('C:/Users/vanholk/sparsifier/dynaphos/config/params.yaml') #TODO move this to our config file
        params['run']['fps'] = 10  # 10 fps -> a single frame represents 100 milliseconds

        n_phosphenes = self.num_phosphenes

        phosphene_coords = cortex_models.get_visual_field_coordinates_probabilistically(params, n_phosphenes)
        #TODO okay so, this initializes a grid with 1000 electrodes (n_phosphenes = 1000), but this grid needs to be the same every iteration
        # so see if i can implement a seed that it it always the same, or another function in cortex_models

        simulator = PhospheneSimulator(params, phosphene_coords)

        print(self.attn_map_soft.shape)
        # img_resized = cv2.resize(self.attn_map_soft, (256, 256))
        # gray_attn = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        #
        # print(gray_attn.shape)

        def normalized_rescaling(img, stimulus_scale=100.e-6):
            """Normalize <img> and rescale the pixel intensities in the range [0, <stimulus_scale>].
            The output image represents the stimulation intensity map.
            param stimulus_scale: the stimulation amplitude corresponding to the highest-valued pixel.
            return: image with rescaled pixel values (stimulation intensity map in Ampères)."""
            img_norm = (img - img.min()) / (img.max() - img.min())
            return img_norm * stimulus_scale

        attn_map_soft_rescaled = normalized_rescaling(self.attn_map_soft)
        stim = simulator.sample_stimulus(attn_map_soft_rescaled)

        # stim = simulator.sample_stimulus(self.attn_map_soft, rescale=True) #stim contains values
        print(stim.shape)

        simulator.reset()
        phosphenes = simulator(stim)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Adjust figsize as needed

        # Plot self.attn_map_soft in the first subplot
        axs[0].imshow(self.attn_map_soft, origin='upper', cmap='gray')
        axs[0].axis('off')
        axs[0].set_title('Attention Map')

        # Plot phosphenes in the second subplot
        axs[1].imshow(phosphenes, origin='upper', cmap='gray')
        axs[1].axis('off')
        axs[1].set_title('Dynaphos Image')

        # Show the plot
        plt.tight_layout()  # Adjust spacing between subplots
        plt.show()

        return phosphenes





    def render_warp(self):
        # if self.opacity_optim:
        #     for group in self.shape_groups:
        #         group.stroke_color.data[:3].clamp_(0., 0.) # to force black stroke
        #         group.stroke_color.data[-1].clamp_(0., 1.) # opacity
                # group.stroke_color.data[-1] = (group.stroke_color.data[-1] >= self.color_vars_threshold).float()


        # if self.add_random_noise:
        #     if random.random() > self.noise_thresh:
        #         eps = 0.01 * min(self.canvas_width, self.canvas_height)
        #         for path in self.shapes:
        #             path.points.data.add_(eps * torch.randn_like(path.points))

        img = self._render()
        return img

    def point_parameters(self):
        # self.points_vars = []
        # points' location optimization
        # for i, point in enumerate(self.point_locations):
        #     if self.optimize_flag[i]:
        #         self.point_locations[i].requires_grad = True
        #         self.points_vars.append(self.point_locations[i])
        self.points_vars = self.point_locations
        self.points_vars.requires_grad = True

        return self.points_vars

    def get_points_params(self):
        return self.points_vars

    # def set_color_parameters(self):
    #     # for storkes' color optimization (opacity)
    #     self.color_vars = []
    #     for i, group in enumerate(self.shape_groups):
    #         if self.optimize_flag[i]:
    #             group.stroke_color.requires_grad = True
    #             self.color_vars.append(group.stroke_color)
    #     return self.color_vars

    # def get_color_parameters(self):
    #     return self.color_vars

    # def save_svg(self, output_dir, name):
    #     pydiffvg.save_svg('{}/{}.svg'.format(output_dir, name), self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
    #

    def save_png(self, output_dir, name):
        canvas_size = (self.canvas_width, self.canvas_height)
        # img = self._render()  # here it is a Tensor
        #
        # # Convert tensor to PIL Image
        # img = transforms.ToPILImage(img)
        img = self._render()
        to_pil = ToPILImage()
        img_pil = to_pil(img)

        img_pil.save('{}/{}.png'.format(output_dir, name), format='PNG', size=canvas_size)

    def dino_attn(self):
        patch_size = 8  # dino hyperparameter
        threshold = 0.6

        # for dino model
        mean_imagenet = torch.Tensor([0.485, 0.456, 0.406])[None,:,None,None].to(self.device)
        std_imagenet = torch.Tensor([0.229, 0.224, 0.225])[None,:,None,None].to(self.device)
        totens = transforms.Compose([
            transforms.Resize((self.canvas_height, self.canvas_width)),
            transforms.ToTensor()
            ])

        dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').eval().to(self.device)

        self.main_im = Image.open(self.target_path).convert("RGB")
        main_im_tensor = totens(self.main_im).to(self.device)
        img = (main_im_tensor.unsqueeze(0) - mean_imagenet) / std_imagenet
        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size

        with torch.no_grad():
            attn = dino_model.get_last_selfattention(img).detach().cpu()[0]

        nh = attn.shape[0]
        attn = attn[:,0,1:].reshape(nh,-1)
        val, idx = torch.sort(attn)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu()

        attn = attn.reshape(nh, w_featmap, h_featmap).float()
        attn = nn.functional.interpolate(attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu()

        return attn

    def define_attention_input(self, target_im):
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        data_transforms = transforms.Compose([
                    preprocess.transforms[-1],
                ])
        self.image_input_attn_clip = data_transforms(target_im).to(self.device)

    def clip_attn(self):
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        text_input = clip.tokenize([self.text_target]).to(self.device)

        if "RN" in self.saliency_clip_model:
            saliency_layer = "layer4"
            attn_map = gradCAM(
                model.visual,
                self.image_input_attn_clip,
                model.encode_text(text_input).float(),
                getattr(model.visual, saliency_layer)
            )
            attn_map = attn_map.squeeze().detach().cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

        else:
            # attn_map = interpret(self.image_input_attn_clip, text_input, model, device=self.device, index=0).astype(np.float32)
            attn_map = interpret(self.image_input_attn_clip, text_input, model, device=self.device)
            plt.imshow(attn_map, interpolation='nearest', vmin=0, vmax=1) #or self.attention_map
            plt.title("atn map")
            plt.axis("off")
            plt.show()

        del model
        return attn_map

    def set_attention_map(self):
        assert self.saliency_model in ["dino", "clip"]
        if self.saliency_model == "dino":
            return self.dino_attn()
        elif self.saliency_model == "clip":
            return self.clip_attn()

    def softmax(self, x, tau=0.2):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum()

    def set_inds_clip(self):
        attn_map = (self.attention_map - self.attention_map.min()) / (self.attention_map.max() - self.attention_map.min())
        if self.xdog_intersec:
            xdog = XDoG_()
            im_xdog = xdog(self.image_input_attn_clip[0].permute(1,2,0).cpu().detach().numpy(), k=10)
            intersec_map = (1 - im_xdog) * attn_map
            attn_map = intersec_map

        attn_map_soft = np.copy(attn_map)
        attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0], tau=self.softmax_temp)

        k = self.num_stages * self.num_phosphenes
        # TODO: change self.stn to output a grid for dynaphos (or whatever it takes, a vector etc.)
        #other_inds = np.random.choice(range(attn_map.flatten().shape[0]), size=k, replace=False, p=attn_map_soft.flatten())
        #other_inds = np.array(np.unravel_index(other_inds, attn_map.shape)).T
        self.inds = self.stn(torch.Tensor(attn_map_soft).unsqueeze(0).unsqueeze(0))

        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] =  self.inds[:, 1].detach() / self.canvas_width
        self.inds_normalised[:, 1] =  self.inds[:, 0].detach() / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()

        plt.imshow(attn_map_soft, interpolation='nearest')  # or self.attention_map
        plt.title("atn map soft")
        plt.axis("off")
        plt.show()

        plt.imshow(intersec_map)
        plt.title("Intersection map")
        plt.axis("off")
        plt.show()

        self.attn_map_soft = attn_map_soft


        return attn_map_soft

    def set_inds_dino(self):
        k = max(3, (self.num_stages * self.num_phosphenes) // 6 + 1) # sample top 3 three points from each attention head
        num_heads = self.attention_map.shape[0]
        self.inds = np.zeros((k * num_heads, 2))
        # "thresh" is used for visualisation purposes only
        thresh = torch.zeros(num_heads + 1, self.attention_map.shape[1], self.attention_map.shape[2])
        softmax = nn.Softmax(dim=1)
        for i in range(num_heads):
            # replace "self.attention_map[i]" with "self.attention_map" to get the highest values among
            # all heads.
            topk, indices = np.unique(self.attention_map[i].numpy(), return_index=True)
            topk = topk[::-1][:k]
            cur_attn_map = self.attention_map[i].numpy()
            # prob function for uniform sampling
            prob = cur_attn_map.flatten()
            prob[prob > topk[-1]] = 1
            prob[prob <= topk[-1]] = 0
            prob = prob / prob.sum()
            thresh[i] = torch.Tensor(prob.reshape(cur_attn_map.shape))

            # choose k pixels from each head
            inds = np.random.choice(range(cur_attn_map.flatten().shape[0]), size=k, replace=False, p=prob)
            inds = np.unravel_index(inds, cur_attn_map.shape)
            self.inds[i * k: i * k + k, 0] = inds[0]
            self.inds[i * k: i * k + k, 1] = inds[1]

        # for visualization
        sum_attn = self.attention_map.sum(0).numpy()
        mask = np.zeros(sum_attn.shape)
        mask[thresh[:-1].sum(0) > 0] = 1
        sum_attn = sum_attn * mask
        sum_attn = sum_attn / sum_attn.sum()
        thresh[-1] = torch.Tensor(sum_attn)

        # sample num_paths from the chosen pixels.
        prob_sum = sum_attn[self.inds[:,0].astype(np.int), self.inds[:,1].astype(np.int)]
        prob_sum = prob_sum / prob_sum.sum()
        new_inds = []
        for i in range(self.num_stages):
            new_inds.extend(np.random.choice(range(self.inds.shape[0]), size=self.num_paths, replace=False, p=prob_sum))
        self.inds = self.inds[new_inds]
        print("self.inds",self.inds.shape)

        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] =  self.inds[:, 1] / self.canvas_width
        self.inds_normalised[:, 1] =  self.inds[:, 0] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()
        return thresh

    def set_attention_threshold_map(self):
        assert self.saliency_model in ["dino", "clip"]
        if self.saliency_model == "dino":
            return self.set_inds_dino()
        elif self.saliency_model == "clip":
            return self.set_inds_clip()

    def get_attn(self):
        return self.attention_map

    def get_thresh(self):
        return self.thresh

    def get_inds(self):
        return self.inds

    def get_mask(self):
        return self.mask

    def set_random_noise(self, epoch):
        if epoch % self.args.save_interval == 0:
            self.add_random_noise = False
        else:
            self.add_random_noise = "noise" in self.args.augmentations


class PainterOptimizer:
    def __init__(self, args, renderer):
        self.renderer = renderer
        self.points_lr = args.lr
        #self.color_lr = args.color_lr
        self.args = args
        #self.optim_color = args.force_sparse

    def init_optimizers(self):
        self.points_optim = torch.optim.Adam([self.renderer.point_parameters()], lr=self.points_lr)
        #if self.optim_color:
         #   self.color_optim = torch.optim.Adam(self.renderer.set_color_parameters(), lr=self.color_lr)

    def update_lr(self, counter):
        # TODO: currently not updating lr, does not do anything
        pass

    def zero_grad_(self):
        self.points_optim.zero_grad()
        #if self.optim_color:
         #   self.color_optim.zero_grad()

    def step_(self):
        self.points_optim.step()
        #if self.optim_color:
         #   self.color_optim.step()

    def get_lr(self):
        return self.points_optim.param_groups[0]['lr']


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)

    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()

    @property
    def activation(self) -> torch.Tensor:
        return self.data

    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad




def interpret(image, texts, model, device):
    images = image.repeat(1, 1, 1, 1)
    res = model.encode_image(images)
    model.zero_grad()
    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)
    cams = [] # there are 12 attention blocks
    for i, blk in enumerate(image_attn_blocks):
        cam = blk.attn_probs.detach() #attn_probs shape is 12, 50, 50
        # each patch is 7x7 so we have 49 pixels + 1 for positional encoding
        cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0)
        cam = cam.clamp(min=0).mean(dim=1) # mean of the 12 something
        cams.append(cam)
        R = R + torch.bmm(cam, R)

    cams_avg = torch.cat(cams) # 12, 50, 50
    cams_avg = cams_avg[:, 0, 1:] # 12, 1, 49
    image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bicubic')
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    return image_relevance


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()

    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:
        # Do a forward and backward pass.
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()

        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)

    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])

    return gradcam


class XDoG_(object):
    def __init__(self):
        super(XDoG_, self).__init__()
        self.gamma=0.98
        self.phi=200
        self.eps=-0.1
        self.sigma=0.8
        self.binarize=True

    def __call__(self, im, k=10):
        if im.shape[2] == 3:
            im = rgb2gray(im)
        imf1 = gaussian_filter(im, self.sigma)
        imf2 = gaussian_filter(im, self.sigma * k)
        imdiff = imf1 - self.gamma * imf2
        imdiff = (imdiff < self.eps) * 1.0  + (imdiff >= self.eps) * (1.0 + np.tanh(self.phi * imdiff))
        imdiff -= imdiff.min()
        imdiff /= imdiff.max()
        if self.binarize:
            th = threshold_otsu(imdiff)
            imdiff = imdiff >= th
        imdiff = imdiff.astype('float32')
        return imdiff
