import os
import random
import argparse

import torch
import numpy as np

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
    parser.add_argument("--target_path", type=str, default="../data/test_set",
                        help="target image path")
    parser.add_argument("--output_path", type=str, default="../output")
    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--mask_object", type=int, default=1)
    parser.add_argument("--fix_scale", type=int, default=0)

    # =================================
    # =========== training ============
    # =================================
    parser.add_argument("--train_set", type=str, default="./data_preprocessed_white_bg/train_set")
    parser.add_argument("--val_set", type=str, default="./data_preprocessed_white_bg/val_set")
    parser.add_argument("--phos_density", type=int, default=100,
                        help="density of phosphenes on the optimized image")
    parser.add_argument("--num_iter", type=int, default=2001,
                        help="number of optimization iterations")
    parser.add_argument("--lr_scheduler", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_duration", type=int, default=10,
                        help="number of epochs during which lr warm up should occur")
    parser.add_argument("--val_step", type=int, default=100,
                        help="model is validated at each val_step")
    parser.add_argument("--epoch_check", type=int, default=10,
                        help="epoch interval to check for model convergence")
    parser.add_argument("--patience", type=int, default=1,
                        help="number of time the convergence criterion should be met")
    parser.add_argument("--delta_val_losses", type=float, default=1e-5,
                        help="threshold for difference between 2 validation losses")
    parser.add_argument("--batch_size_training", type=int, default=8)
    parser.add_argument("--batch_size_validation", type=int, default=8)
    parser.add_argument("--image_scale", type=int, default=224)
    parser.add_argument("--num_img_tracked", type=int, default=2,
                        help="number of validation images tracked during training")

    # =====================================
    # ======== contours extraction ========
    # =====================================
    parser.add_argument("--sigma_kernel", type=float, default=2,
                        help="Size of sigma kernel for contours extraction")
    parser.add_argument("--lambda_kernel", type=float, default=4,
                        help="Size of lambda kernel for contours extraction")
    parser.add_argument("--padding_pix", type=int, default=3)
    parser.add_argument("--padding_color", type=int, default=1)
    parser.add_argument("--n_orientations", type=int, default=8,
                        help="Number of orientations to extract for contours")

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
    parser.add_argument("--augemntations", type=str, default="affine",
                        help="can be any combination of: 'affine_noise_eraserchunks_eraser_press'")
    parser.add_argument("--noise_thresh", type=float, default=0.5)
    parser.add_argument("--aug_scale_min", type=float, default=0.7)
    parser.add_argument("--force_sparse", type=float, default=0,
                        help="if True, use L1 regularization on stroke's opacity to encourage small number of strokes")
    parser.add_argument("--clip_conv_loss", type=float, default=1)
    parser.add_argument("--clip_conv_loss_type", type=str, default="L2")
    parser.add_argument("--clip_conv_layer_weights",
                        type=str, default="0,0,1.0,1.0,0")
    parser.add_argument("--clip_model_name", type=str, default="RN101")
    parser.add_argument("--clip_fc_loss_weight", type=float, default=0.1)
    parser.add_argument("--clip_text_guide", type=float, default=0)
    parser.add_argument("--text_target", type=str, default="none")
    parser.add_argument("--penalization_weight", type=float, default=1.0)

    args = parser.parse_args()
    set_seed(args.seed)

    args.clip_conv_layer_weights = [
        float(item) for item in args.clip_conv_layer_weights.split(',')]

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    if args.use_gpu:
        args.device = torch.device("cuda" if (
            torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    else:
        args.device = torch.device("cpu")
    return args


if __name__ == "__main__":
    # for cog predict
    args = parse_arguments()
    final_config = vars(args)
    np.save(f"{args.output_path}/config_init.npy", final_config)
