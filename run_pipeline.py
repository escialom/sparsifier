import torch

import data_preprocessing as preprocess
from config import model_config
from training_model import train_model

args = model_config.parse_arguments()

# Data Preprocessing: object segmentation
if args.preprocess_data is True:
    preprocess.extract_background(args, args.train_set, args.output_dir)

# Training and validating model
if args.train_mode is True:
    trained_model = train_model(args)
    torch.save(trained_model, f"{args.output_dir}/model.pth")
    # Check results of training and validation

# Test model: generate the stimuli (experimental condition)
