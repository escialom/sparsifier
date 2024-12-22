import os

import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import dynaphos
import utils
import data_preprocessing as preprocess
from config import model_config
from model import PhospheneOptimizer, InitMap
from training_model import train_model
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator

preprocess_data = False
train_mode = False
test_mode = False

args = model_config.parse_arguments()
simulator_params = dynaphos.utils.load_params("./config/config_dynaphos/params.yaml")

# Data Preprocessing: object segmentation
if preprocess_data:
    preprocess.extract_background(args, args.train_set, args.output_dir)

# Training and validating model
if train_mode:
    model_weights = train_model(args)
    torch.save(model_weights, f"{args.output_dir}/model_weights.pth")
    # Check results of training and validation
    utils.plot_losses("./output", "training_data_checkpoints_200phos.pth", "validation_data_checkpoints_200phos.pth")

if test_mode:
    # Test model: generate the stimuli (experimental condition)
    weights = torch.load("./output/model_weights.pth", map_location=torch.device('cpu'))
    model = PhospheneOptimizer(args=args,
                               simulator_params=simulator_params,
                               electrode_grid=1024,
                               batch_size=1,
                               n_phos=args.num_phos)
    model.load_state_dict(weights)
    model.eval()
    transform = transforms.Compose([transforms.Resize((args.image_scale, args.image_scale)),
                                    transforms.CenterCrop(args.image_scale),
                                    transforms.ToTensor()])
    test_dataset = ImageFolder(root="./data/test_set", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for batch_idx, (batch, _) in enumerate(test_loader):
        for img_idx, img_sample in enumerate(batch):
            img_sample = img_sample.unsqueeze(0)
            with torch.no_grad():
                output_imgs, _ = model(img_sample)
                #utils.plot_optimized_img(output_imgs)
                relative_img_path, _ = test_loader.dataset.samples[batch_idx * len(batch) + img_idx]
                input_filename = os.path.splitext(os.path.basename(relative_img_path))[0]
                val_img_post_training_dir = os.path.join(args.output_dir, "post_training_test_imgs")
                output_img_dir = os.path.join(val_img_post_training_dir, os.path.dirname(relative_img_path))
                os.makedirs(output_img_dir, exist_ok=True)
                utils.save_images(output_imgs, output_img_dir, input_filename, epoch_idx=args.num_iter)


# Control images
transform = transforms.Compose([transforms.Resize((args.image_scale, args.image_scale)),
                                    transforms.CenterCrop(args.image_scale),
                                    transforms.ToTensor()])
test_dataset = ImageFolder(root="./data/test_set", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Prepare the phosphene simulator for control stimuli (contours and brightness)
phosphene_coords = dynaphos.cortex_models.get_visual_field_coordinates_probabilistically(simulator_params, 1024, use_seed=True)
simulator = PhospheneSimulator(simulator_params, phosphene_coords, batch_size=1, n_phos=args.num_phos)
# Generate the control images (contour and brightness stimuli)
contours_extraction = InitMap(args)
for batch_idx, (batch, _) in enumerate(test_loader):
    for img_idx, img_sample in enumerate(batch):
        img_sample = img_sample.unsqueeze(0)
        # Contours
        contour_img = contours_extraction(img_sample, requires_grad=False)
        contour_norm = (contour_img - contour_img.min()) / (contour_img.max() - contour_img.min())
        contour_norm *= simulator_params['sampling']['stimulus_scale']
        phosphene_placement_contours = simulator.sample_stimulus(contour_norm)
        simulator.reset()
        contour_phos, stim_intensity_cont = simulator(phosphene_placement_contours)
        relative_img_path, _ = test_loader.dataset.samples[batch_idx * len(batch) + img_idx]
        input_filename = os.path.splitext(os.path.basename(relative_img_path))[0]
        val_img_post_training_dir = os.path.join(args.output_dir, "contour_stim")
        output_img_dir = os.path.join(val_img_post_training_dir, os.path.dirname(relative_img_path))
        os.makedirs(output_img_dir, exist_ok=True)
        filename_output = f"{input_filename}_contour_phos_{args.num_iter}.png"
        plt.imsave(os.path.join(output_img_dir, filename_output), contour_phos.squeeze().numpy(), cmap='gray')
        #utils.save_images(contour_phos, output_img_dir, input_filename, epoch_idx=args.num_iter)
        # Brightness
        img_norm = 1 - ((img_sample - img_sample.min()) / (img_sample.max() - img_sample.min()))
        img_norm = transforms.Grayscale()(img_norm)
        img_norm *= simulator_params['sampling']['stimulus_scale']
        phosphene_placement_brightness = simulator.sample_stimulus(img_norm)
        simulator.reset()
        brightness_phos, stim_intensity_bright = simulator(phosphene_placement_brightness)
        input_filename = os.path.splitext(os.path.basename(relative_img_path))[0]
        val_img_post_training_dir = os.path.join(args.output_dir, "brightness_stim")
        output_img_dir = os.path.join(val_img_post_training_dir, os.path.dirname(relative_img_path))
        os.makedirs(output_img_dir, exist_ok=True)
        filename_output = f"{input_filename}_brightness_phos_{args.num_iter}.png"
        plt.imsave(os.path.join(output_img_dir, filename_output), brightness_phos.squeeze().numpy(), cmap='gray')











