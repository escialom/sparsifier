import argparse
import os
import random

import numpy as np
import torch

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
    parser.add_argument("--target", help="target image path")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--use_gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--mask_object", type=int, default=1)
    parser.add_argument("--fix_scale", type=int, default=0)

    # =================================
    # =========== training ============
    # =================================
    parser.add_argument("--train_set", type=str, default="./data/train_set")
    parser.add_argument("--val_set", type=str, default="./data/val_set")
    parser.add_argument("--num_iter", type=int, default=3,
                        help="number of optimization iterations")
    parser.add_argument("--lr_scheduler", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size_training", type=int, default=2)
    parser.add_argument("--batch_size_validation", type=int, default=2)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--image_scale", type=int, default=224)

    # =================================
    # ======== map init params ========
    # =================================
    parser.add_argument("--attention_init", type=int, default=1,
                        help="if True, use the attention heads of Dino model to set the location of the initial strokes")
    parser.add_argument("--saliency_model", type=str, default="clip")
    parser.add_argument("--saliency_clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--xdog_intersec", type=int, default=1)
    parser.add_argument("--init_mode", type=str, default="contours")
    parser.add_argument("--mask_object_attention", type=int, default=0)
    parser.add_argument("--softmax_temp", type=float, default=0.3)

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
    parser.add_argument("--clip_model_name", type=str, default="RN50") # default="RN101"
    parser.add_argument("--clip_fc_loss_weight", type=float, default=0.1)
    parser.add_argument("--clip_text_guide", type=float, default=0)
    parser.add_argument("--text_target", type=str, default="none")

    args = parser.parse_args()
    set_seed(args.seed)

    args.clip_conv_layer_weights = [
        float(item) for item in args.clip_conv_layer_weights.split(',')]

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

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
    np.save(f"{args.output_dir}/config_init.npy", final_config)