import os

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import utils
import dynaphos
from model import PhospheneOptimizer
import config.model_config as model_config

check_img = False
plot = False
check_params = False

# training_data = torch.load("./output/training_data_checkpoints.pth", map_location=torch.device('cpu'))
# training_data[210]['epoch_loss']

args = model_config.parse_arguments()
weights = torch.load("./output/model.pth", map_location=torch.device('cpu'))
model = PhospheneOptimizer(args=args,
                           simulator_params=dynaphos.utils.load_params("./config/config_dynaphos/params.yaml"),
                           electrode_grid=1024,
                           batch_size=1,
                           n_phos=200)
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
            utils.save_images(output_imgs, output_img_dir, input_filename, epoch_idx=400)


if plot:
    utils.plot_losses("./output", "training_data_checkpoints.pth", "validation_data_checkpoints.pth")
    #utils.plot_stim_properties("./output", "data_tracked_validation_img.pth")
    #utils.val_img_properties("./output", "data_val_imgs_after_training.pth")

if check_img:
    # Load model
    args = model_config.parse_arguments()
    epoch = 1
    training_data = torch.load("./output/training_data_checkpoints.pth", map_location=torch.device('cpu'))
    model = PhospheneOptimizer(args=args,
                                   simulator_params=dynaphos.utils.load_params("./config/config_dynaphos/params.yaml"),
                                   electrode_grid=1024,
                                   batch_size=args.batch_size_training)
    model.load_state_dict(training_data[epoch]['model_state_dict'])
    model.load_state_dict(training_data[10]['model_state_dict'])
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
                val_img_post_training_dir = os.path.join(args.output_dir, "post_training_training")
                output_img_dir = os.path.join(val_img_post_training_dir, os.path.dirname(relative_img_path))
                os.makedirs(output_img_dir, exist_ok=True)
                utils.save_images(output_imgs, output_img_dir, input_filename, epoch)

if check_params:
    args = model_config.parse_arguments()
    training_data = torch.load("./output/training_data_checkpoints.pth", map_location=torch.device('cpu'))
    model = PhospheneOptimizer(args=args,
                               simulator_params=dynaphos.utils.load_params("./config/config_dynaphos/params.yaml"),
                               electrode_grid=1024,
                               batch_size=args.batch_size_training)
    for epoch in range(22):
        # Load model parameters for the specified epoch
        model.load_state_dict(training_data[epoch]['model_state_dict'])

        print(f"Parameters at Epoch {epoch}:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name} | Weights Mean: {param.data.mean().item()} | Weights Std: {param.data.std().item()}")
