import argparse
import os
import random

import numpy as np
import torch
from torch import Tensor
# import wandb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # =================================
    # ============ general ============
    # =================================
    abs_path = os.path.abspath(os.getcwd())
    target = f"{abs_path}/target_images/camel.png"  # These lines gave wrong formatting
    assert os.path.isfile(target), f"{target} does not exist!"
    test_name = os.path.splitext("camel.png")[0]
    output_dir = f"{abs_path}/output_sketches/{test_name}/"  # this line gave wrong formatting
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    parser.add_argument("--target",default= f"{abs_path}/target_images/camel.png", help="target image path")
    parser.add_argument("--target_file", type=str, default="camel.png",
                        help="target image file, located in <target_images>")
    parser.add_argument("--output_dir", type=str, default=f"output_sketches/{test_name}/",
                        help="directory to save the output images and loss")
    parser.add_argument("--path_svg", type=str, default="none",
                        help="if you want to load an svg file and train from it")
    parser.add_argument("--use_gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask_object", type=int, default=0, help="if the target image contains background, it's better to mask it out")
    parser.add_argument("--fix_scale", type=int, default=0, help="if the target image is not squared, it is recommended to fix the scale")
    parser.add_argument("--display_logs", type=int, default=0)
    parser.add_argument("--display", type=int, default=0)
    parser.add_argument("--multiprocess", type=int, default=0,
                        help="recommended to use multiprocess if your computer has enough memory")
    parser.add_argument('-colab', action='store_true')
    parser.add_argument('-cpu', action='store_true')
    parser.add_argument('-display', action='store_true')
    parser.add_argument('--gpunum', type=int, default=0)

    # =================================
    # ============ wandb ============
    # =================================
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--wandb_user", type=str, default="yael-vinker")
    parser.add_argument("--wandb_name", type=str, default="test")
    parser.add_argument("--wandb_project_name", type=str, default="none")

    # =================================
    # =========== training ============
    # =================================
    parser.add_argument("--num_iter", type=int, default=2001,
                        help="number of optimization iterations") #default = 500
    parser.add_argument("--num_stages", type=int, default=1,
                        help="training stages, you can train x strokes, then freeze them and train another x strokes etc.")
    parser.add_argument("--num_sketches", type=int, default=3,
                        help="it is recommended to draw 3 sketches and automatically chose the best one")
    parser.add_argument("--lr_scheduler", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0) #default = 1.0
    #parser.add_argument("--color_lr", type=float, default=0.01)
    #parser.add_argument("--color_vars_threshold", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="for optimization it's only one image")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--image_scale", type=int, default=224)
    parser.add_argument('--canvas', type=Tensor,
                        help='Output image size. This should match to downstream model input size.',
                        default=torch.zeros([224, 224]))

    # =================================
    # ======== phosphene params =========
    # =================================
    parser.add_argument("--num_phosphenes", type=int,
                        default=1000, help="number of phosphenes used to generate the image, this defines the level of density.") #TODO: num_phosphenes
    #parser.add_argument("--width", type=float,
    #                    default=1.5, help="stroke width")
    #parser.add_argument("--control_points_per_seg", type=int, default=4)
    #parser.add_argument("--num_segments", type=int, default=1,
    #                    help="number of segments for each stroke, each stroke is a bezier curve with 4 control points")
    parser.add_argument('--patch_size', type=int, help='Size of each of the patches of phosphenes.', default=8) #10
    parser.add_argument('--phosphene_radius', type=int, help='Radius of the phosphene.', default=1.2) #1.5
    parser.add_argument("--attention_init", type=int, default=1,
                        help="if True, use the attention heads of Dino model to set the location of the initial strokes")
    parser.add_argument("--saliency_model", type=str, default="clip")
    parser.add_argument("--saliency_clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--xdog_intersec", type=int, default=1)
    parser.add_argument("--mask_object_attention", type=int, default=0)
    parser.add_argument("--softmax_temp", type=float, default=0.3)
    parser.add_argument("--constrain", type=int, default= 1)

    # =================================
    # ============= loss ==============
    # =================================
    parser.add_argument("--percep_loss", type=str, default="none",
                        help="the type of perceptual loss to be used (L2/LPIPS/none)")
    parser.add_argument("--perceptual_weight", type=float, default=0,
                        help="weight the perceptual loss")
    parser.add_argument("--train_with_clip", type=int, default=0)
    parser.add_argument("--clip_weight", type=float, default=0)
    parser.add_argument("--start_clip", type=int, default=0)
    parser.add_argument("--num_aug_clip", type=int, default=4)
    parser.add_argument("--include_target_in_aug", type=int, default=0)
    parser.add_argument("--augment_both", type=int, default=1,
                        help="if you want to apply the affine augmentation to both the sketch and image")
    parser.add_argument("--augmentations", type=str, default="affine",
                        help="can be any combination of: 'affine_noise_eraserchunks_eraser_press'")
    parser.add_argument("--noise_thresh", type=float, default=0.5)
    parser.add_argument("--aug_scale_min", type=float, default=0.7)
    #parser.add_argument("--force_sparse", type=float, default=0,
    #                   help="if True, use L1 regularization on stroke's opacity to encourage small number of strokes")
    parser.add_argument("--clip_conv_loss", type=float, default=1)
    parser.add_argument("--clip_conv_loss_type", type=str, default="L2")
    parser.add_argument("--clip_conv_layer_weights",
                        type=str, default="0,0,1.0,1.0,0")
    parser.add_argument("--clip_model_name", type=str, default="RN50")
    parser.add_argument("--clip_fc_loss_weight", type=float, default=0.1)
    parser.add_argument("--clip_text_guide", type=float, default=0)
    parser.add_argument("--text_target", type=str, default="none")

    args = parser.parse_args()
    set_seed(args.seed)

    assert args.image_scale % args.patch_size == 0 #TODO change patch_size to 8 or 7? and change phosphene radius as well accordingly on the same scale?

    args.clip_conv_layer_weights = [
        float(item) for item in args.clip_conv_layer_weights.split(',')]

    args.output_dir = os.path.join(args.output_dir, args.wandb_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    jpg_logs_dir = f"{args.output_dir}/jpg_logs"
    png_logs_dir = f"{args.output_dir}/png_logs"
    if not os.path.exists(jpg_logs_dir):
        os.mkdir(jpg_logs_dir)
    if not os.path.exists(png_logs_dir):
        os.mkdir(png_logs_dir)

    # if args.use_wandb:
    #     wandb.init(project=args.wandb_project_name, entity=args.wandb_user,
    #                config=args, name=args.wandb_name, id=wandb.util.generate_id())

    if args.use_gpu:
        args.device = torch.device("cuda" if (
            torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    else:
        args.device = torch.device("cpu")
    # pydiffvg.set_use_gpu(torch.cuda.is_available() and args.use_gpu) #TODO: edit this
    # pydiffvg.set_device(args.device) #TODO: edit this
    return args


if __name__ == "__main__":
    # for cog predict
    args = parse_arguments()
    final_config = vars(args)
    np.save(f"{args.output_dir}/config_init.npy", final_config)