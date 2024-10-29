import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image, to_tensor

import dynaphos
import utils
import config.model_config as model_config
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator
from model import InitMap

# Load params
args = model_config.parse_arguments()
simulator_params = dynaphos.utils.load_params("./config/config_dynaphos/params.yaml")
phosphene_coords = dynaphos.cortex_models.get_visual_field_coordinates_probabilistically(simulator_params,
                                                                                         n_phosphenes=1024,
                                                                                         use_seed=True)
# Init phosphene simulator and contour's extraction algorithm
simulator = PhospheneSimulator(params=simulator_params, coordinates=phosphene_coords, batch_size=1)
if args.ctrl_stim_mode == "contours":
    get_contours = InitMap(args)

#Load test images
if args.ctrl_stim_mode == "contours":
    transform = transforms.Compose([transforms.Resize((args.image_scale, args.image_scale)),
                                transforms.CenterCrop(args.image_scale),
                                transforms.ToTensor()])
elif args.ctrl_stim_mode == "brightness":
    transform = transforms.Compose([transforms.Resize((args.image_scale, args.image_scale)),
                                    transforms.CenterCrop(args.image_scale),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor()])
test_dataset = ImageFolder(root="./data/black_bg", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

decrease = False

for batch in test_loader:
    input_imgs, _ = batch
    input_imgs[input_imgs == 0] = 1 # background is white
    if args.ctrl_stim_mode == "contours":
        contours = get_contours(input_imgs)
        out = contours.squeeze(0).squeeze(0)
        numpy_img = out.detach().numpy()
        numpy_img = (numpy_img - numpy_img.min()) / (numpy_img.max() - numpy_img.min())
        plt.imshow(numpy_img, cmap='gray')
        plt.axis('off')  # Turn off axis labels
        plt.show()
        phosphene_placement_map = utils.normalized_rescaling(contours, max_stimulation_intensity=simulator_params['sampling']['stimulus_scale'])
    elif args.ctrl_stim_mode == "brightness":
        phosphene_placement_map = utils.normalized_rescaling(input_imgs.squeeze(), max_stimulation_intensity=simulator_params['sampling']['stimulus_scale'])
        phosphene_placement_map = phosphene_placement_map.unsqueeze(0).unsqueeze(0)
    phosphene_placement_map = simulator.sample_stimulus(phosphene_placement_map)

    if decrease: # TODO: Add a for loop for the 9 phosphene densities
        topk_values, topk_indices = torch.topk(phosphene_placement_map, k=100, dim=2)
        result_tensor = torch.zeros_like(phosphene_placement_map)
        result_tensor.scatter_(2, topk_indices, topk_values)
        simulator.reset()
        optimized_im, stim_intensity = simulator(result_tensor)
    else:
        simulator.reset()
        optimized_im, stim_intensity = simulator(phosphene_placement_map)

    optimized_im = optimized_im.squeeze(0)
    optimized_im = optimized_im.numpy()
    optimized_im = (optimized_im - optimized_im.min()) / (optimized_im.max() - optimized_im.min())
    plt.imshow(optimized_im, cmap='gray')
    plt.axis('off')  # Turn off axis labels
    plt.show()