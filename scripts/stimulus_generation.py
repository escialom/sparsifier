"""
Generate matched phosphene stimuli across multiple image-generation conditions.

This script loads a trained phosphene optimization model and, for each input
image, generates three types of phosphene stimuli:

1. DNN-optimized stimuli produced by the trained phosphene optimizer
2. Contour-based stimuli
3. Luminance-based (grayscale) stimuli

For each image, the number of active electrodes is computed for all three
conditions. The condition with the lowest number of activated electrodes is
used as the reference condition. The placement maps of the other two
conditions are then reduced so that all three conditions are matched to the
same number of electrodes. The reduced placement maps are subsequently
re-phosphenized with `reset_thresholds=True` to ensure a consistent final
stimulus generation step.

Finally, the script:
- stores the matched number of phosphenes in a CSV file,
- reconstructs the appropriate relative output path,
- saves the three matched phosphene images to disk.

This procedure ensures that phosphene stimuli generated from different image
representations are equated in terms of electrode count, enabling controlled
comparisons across conditions.

Parameters
----------
args : argparse.Namespace
    Configuration object parsed from `model_config.parse_arguments()`.
    Expected attributes include at least:
    - image_scale : target resize dimension for input images
    - target_path : root directory containing input images
    - output_path : directory for loading model checkpoints and saving outputs
    - phos_density : phosphene density condition used in stimulus generation

Returns
-------
None
"""

import torch
from torchvision import transforms
from pathlib import Path

import dynaphos
import config.model_config as model_config
from src.model import PhospheneOptimizer

from src.StimGen import DataLoading
from src.StimGen import StimulusRendering
from src.StimGen import SavingStimuli


# Load arguments and get dataloader
args = model_config.parse_arguments()
transform = transforms.Compose([transforms.Resize((args.image_scale, args.image_scale)), transforms.ToTensor()])
data_manager = DataLoading(args.target_path,
                           args.output_path,
                           transform=transform)
loader, dataset = data_manager.get_data_loader(batch_size=1)

# Load Simulator params
simulator_params = dynaphos.utils.load_params("../config/config_dynaphos/params.yaml")
phosphene_coordinates = dynaphos.cortex_models.get_visual_field_coordinates_probabilistically(simulator_params,
                                                                                                  n_phosphenes=1024,
                                                                                                  use_seed=True)
# Load trained model
model = PhospheneOptimizer(args=args,
                           simulator_params=simulator_params,
                           electrode_grid=1024,
                           batch_size=1,
                           phos_density=args.phos_density)
weights = torch.load(f"{args.output_path}/checkpoint_epoch_416.pth", map_location=torch.device('cpu'))
weights = weights['model_state_dict']
model.load_state_dict(weights)
model.eval()

# Load Stimulus Generator
StimulusRenderer = StimulusRendering(args,
                                     batch_size=1,
                                     model=model,
                                     simulator_params=simulator_params,
                                     phosphene_coordinates=phosphene_coordinates)

# Load Saving class
save_stimuli = SavingStimuli(args.output_path)
save_raw_contours = SavingStimuli(Path(args.output_path) / "raw_contours")
save_raw_luminance = SavingStimuli(Path(args.output_path) / "raw_luminance")

# Stimulus generation loop
for batch_idx, (batch, _) in enumerate(loader):
    for img_idx, input_img in enumerate(batch):
        input_img = input_img.unsqueeze(0)
        # Get the relative path to the input file
        relative_path, _ = loader.dataset.samples[batch_idx * len(batch) + img_idx]
        relative_path = Path(relative_path)
        # Get DNN stimuli
        phos_DNN, n_elecs_DNN, current_DNN, DNN_map = StimulusRenderer.get_optimized_stimuli(input_img)
        # Get contours stimuli
        contours = StimulusRenderer.get_contours(input_img)
        phos_contours, n_elecs_cont, current_cont, cont_map = StimulusRenderer.phosphenize(contours)
        # Save contours
        contour_output_dir = save_raw_contours.get_relative_output_dir(relative_path.parent, dataset)
        save_raw_contours.save_stimuli(contours.squeeze(),
                                       output_filename=f"{relative_path.stem}_contours_raw.png",
                                       output_img_dir=contour_output_dir)
        # Get luminance stimuli
        gray_img = StimulusRenderer.get_luminance(input_img)
        phos_lum, n_elecs_lum, current_lum, lum_map = StimulusRenderer.phosphenize(gray_img)
        # Save luminance
        luminance_output_dir = save_raw_luminance.get_relative_output_dir(relative_path.parent, dataset)
        save_raw_luminance.save_stimuli(gray_img.squeeze(),
                                        output_filename=f"{relative_path.stem}_luminance_raw.png",
                                        output_img_dir=luminance_output_dir)
        # Create dict of stimuli
        stimulus_dict = {"DNN": {"num_elecs": n_elecs_DNN,
                                 "phos_img": phos_DNN,
                                 "current":current_DNN,
                                 "map": DNN_map,},
                         "contours": {"num_elecs": n_elecs_cont,
                                      "phos_img": phos_contours,
                                      "current":current_cont,
                                      "map": cont_map,},
                         "luminance": {"num_elecs": n_elecs_lum,
                                       "phos_img": phos_lum,
                                       "current": current_lum,
                                       "map": lum_map,}}
        # Identify the placement map with the lowest number of electrodes
        sorted_cond = StimulusRenderer.sort_num_elecs(stimulus_dict)
        primary = sorted_cond[0]
        secondary = sorted_cond[1]
        tertiary = sorted_cond[2]
        # Define condition with lowest number of phosphenes as target img
        target_img = stimulus_dict[primary]["phos_img"]
        target_num_elecs = stimulus_dict[primary]["num_elecs"]
        current_target = stimulus_dict[primary]["current"]
        # Assign the other conditions to be matched
        current_map_one = stimulus_dict[secondary]["map"]
        current_map_two = stimulus_dict[tertiary]["map"]
        # Match the number of elecs to the condition having the lowest number of elecs
        reduced_map_one = StimulusRenderer.filter_num_elecs(current_map_one, target_num_elecs=target_num_elecs)
        reduced_map_two = StimulusRenderer.filter_num_elecs(current_map_two, target_num_elecs=target_num_elecs)
        img_phos_one, num_elecs_one, sim_current_one, _ = StimulusRenderer.phosphenize(reduced_map_one, reset_thresholds=True)
        img_phos_two, num_elecs_two, sim_current_two, _ = StimulusRenderer.phosphenize(reduced_map_two, reset_thresholds=True)
        print(num_elecs_one, target_num_elecs)
        print(num_elecs_two, target_num_elecs)
        # Creates csv file with number of phosphenes
        save_stimuli.save_num_phos(relative_path, args.phos_density, target_num_elecs)
        output_img_dir = save_stimuli.get_relative_output_dir(relative_path.parent, dataset)
        # Save img with lowest number of phos (target img)
        save_stimuli.save_stimuli(target_img.squeeze(),
                                  output_filename= f"{relative_path.stem}_{primary}_{args.phos_density}.png",
                                  output_img_dir=output_img_dir)
        # Save img with 2nd lowest number of phos
        save_stimuli.save_stimuli(img_phos_one.squeeze(),
                                  output_filename=f"{relative_path.stem}_{secondary}_{args.phos_density}.png",
                                  output_img_dir=output_img_dir)
        # Save img with highest number of phos
        save_stimuli.save_stimuli(img_phos_two.squeeze(),
                                  output_filename=f"{relative_path.stem}_{tertiary}_{args.phos_density}.png",
                                  output_img_dir=output_img_dir)