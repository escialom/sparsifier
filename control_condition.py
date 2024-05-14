import os
from pathlib import Path

import cv2
import torch
import torchvision
from PIL import Image

import PIL
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage

from dynaphos import simulator, utils, cortex_models
from dynaphos.image_processing import sobel_processor
import config
from simplified_painter import normalized_rescaling, get_target_and_mask
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator

args = config.parse_arguments()

def get_control_im(args, stimulus_scale):
    # Load parameters
    params = utils.load_params('C:/Users/vanholk/sparsifier/dynaphos/config/params.yaml')

    # Get phosphene coordinates
    phosphene_coords = cortex_models.get_visual_field_coordinates_probabilistically(params, args.num_phosphenes)

    # Initialize simulator
    simulator = PhospheneSimulator(params, phosphene_coords)

    # Get target image and mask
    abs_path = Path(os.path.abspath(os.getcwd()))
    image_paths = Path(f"{abs_path}/target_images/horse.png")
    target_im, mask = get_target_and_mask(args, target_image_path=image_paths)

    # Process target image
    target_im = target_im[0, 0, :, :].numpy()
    img_sobel = sobel_processor(target_im)
    phosphene_placement_map = normalized_rescaling(img_sobel, stimulus_scale=stimulus_scale)

    # Sample stimulus
    phosphene_placement_map = simulator.sample_stimulus(phosphene_placement_map, rescale=False)
    simulator.reset()

    # Generate control image
    control_im = simulator(phosphene_placement_map)

    return control_im



