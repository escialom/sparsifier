import torchvision.transforms as transforms
import os
from pathlib import Path
import torchvision
from PIL import Image
from dynaphos import utils, cortex_models
from dynaphos.image_processing import sobel_processor
import config
from simplified_painter import normalized_rescaling, get_target_and_mask
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator

abs_path = os.path.abspath(os.getcwd())
args = config.parse_arguments()
params = utils.load_params(f"{abs_path}/dynaphos/config/params.yaml")

# Get target image and mask
#todo make argument for name of target for names of files
image_paths = Path(f"{abs_path}/target_images/horse.png")
target_im, mask = get_target_and_mask(args, target_image_path=image_paths)

phosphene_coords = cortex_models.get_visual_field_coordinates_probabilistically(params, args.electrode_grid)
simulator = PhospheneSimulator(params, phosphene_coords, args.top_k_values)

target_im = target_im[0, 0, :, :].numpy()
img_sobel = sobel_processor(target_im)
phosphene_placement_map = normalized_rescaling(img_sobel, stimulus_scale=args.stimulus_scale_control)
phosphene_placement_map = simulator.sample_stimulus(phosphene_placement_map, rescale=False)
simulator.reset()

# Generate control image
control_im = simulator(phosphene_placement_map)
output_path_control = f"{args.output_dir}/control_im.png"
torchvision.utils.save_image(control_im, output_path_control)


#Postprocessing
# Load the image
optimized_im = Image.open(f"{abs_path}/output_sketches/horse/test/best_iter.png")
control_im = Image.open(f"{abs_path}/output_sketches/horse/test/control_im.png")
transform = transforms.ToTensor()
optimized_im = transform(optimized_im)
control_im = transform(control_im)

should_process = True  # Flag to control processing
if optimized_im.sum() / control_im.sum() <=0.8:
    reference_img = control_im.clone()
    target_img = optimized_im.clone()
    output_path_rescaled = f"{args.output_dir}/rescaled_opt_im.png"

elif optimized_im.sum() / control_im.sum() >= 1.2:
    reference_img = optimized_im.clone()
    target_img = control_im.clone()
    output_path_rescaled = f"{args.output_dir}/rescaled_cont_im.png"

else:
    control_im = control_im
    optimized_im = optimized_im
    print("No rescaling is necessary. The saved control image and best_iter are the final images")
    should_process = False  # Set flag to False to skip further processin


if should_process:  # Only execute this block if the flag is True
    target_img_power = target_img ** args.exp_postprocessing
    target_img_power[target_img_power == 0.] = 1e-6
    reference_img_power = reference_img ** args.exp_postprocessing
    reference_img_power[reference_img_power == 0.] = 1e-6
    ratio_power = reference_img_power.sum() / target_img_power.sum()
    target_img_rescaled = target_img * ratio_power

    torchvision.utils.save_image(target_img_rescaled, output_path_rescaled)

    # Combine both images side by side using make_grid
    combined_images = torchvision.utils.make_grid([reference_img, target_img_rescaled], nrow=2, padding=10,
                                                  normalize=False, scale_each=False, pad_value=0)

    output_path_combined = f"{args.output_dir}/comparison_control_optimized.png"
    torchvision.utils.save_image(combined_images, output_path_combined)


