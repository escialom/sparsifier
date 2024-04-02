import warnings
import argparse
import math
import os
import sys
import time
import traceback

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess as sp
# import wandb
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from tqdm.auto import tqdm, trange

import config
import sketch_utils as utils
from models.loss import Loss
from models.painter_params import Painter, PainterOptimizer
from IPython.display import display, SVG
from models.loss import CLIPConvLoss

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


def load_renderer(args, target_im=None, mask=None):
    renderer = Painter(num_phosphenes=args.num_phosphenes, args=args,
                       # num_segments=args.num_segments,
                       imsize=args.image_scale,
                       device=args.device,
                       target_im=target_im,
                       mask=mask)
    renderer = renderer.to(args.device)
    return renderer


def get_target(args): # moved to the new Painter model
    target = Image.open(args.target)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    masked_im, mask = utils.get_mask_u2net(args, target)
    if args.mask_object:
        target = masked_im
    if args.fix_scale:
        target = utils.fix_image_scale(target)

    transforms_ = []
    if target.size[0] != target.size[1]:
        transforms_.append(transforms.Resize(
            (args.image_scale, args.image_scale), interpolation=PIL.Image.BICUBIC))
    else:
        transforms_.append(transforms.Resize(
            args.image_scale, interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(args.image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    target_ = data_transforms(target).unsqueeze(0).to(args.device)
    return target_, mask


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def plotting(epoch):
#     final_config = np.load(f"{args.output_dir}/config.npy", allow_pickle=True).item()
#
#     # Access the loss_eval list
#     loss_eval = final_config['loss_eval']
#
#     epochs = list(range(1, len(loss_eval) + 1))
#     # Plotting
#     plt.figure(figsize=(10, 6))  # Set the figure size for better readability
#     fig, ax = plt.subplots()
#     ax.plot(epochs, loss_eval, label='Loss', marker='o', linestyle='-')
#     fig.title('Loss over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f"{args.output_dir}/loss_{epoch}.png")



def main(args):
    loss_func = Loss(args)
    inputs, mask = get_target(args)
    utils.log_input(args.use_wandb, 0, inputs, args.output_dir)
    renderer = load_renderer(args, inputs, mask)
    print(count_trainable_parameters(renderer))

    optimizer = PainterOptimizer(args, renderer)
    counter = 0
    configs_to_save = {"loss_eval": []}
    best_loss = 100
    best_fc_loss = 100
    best_iter, best_iter_fc = 0, 0
    min_delta = 1e-5
    terminate = False

    renderer.set_random_noise(0)
    img = renderer.init_image(stage=0)
    optimizer.init_optimizers()

    # not using tdqm for jupyter demo
    if args.display:
        epoch_range = range(args.num_iter)
    else:
        epoch_range = tqdm(range(args.num_iter))

    for epoch in epoch_range:
        if not args.display:
            epoch_range.refresh()
        renderer.set_random_noise(epoch)
        if args.lr_scheduler:
            optimizer.update_lr(counter)

        start = time.time()
        optimizer.zero_grad_()
        sketches = renderer.get_image().to(args.device)  # For every epoch it gets sketches, which is every time the newly initialized image

        losses_dict = loss_func(sketches, inputs.detach(
        ), renderer.get_activation_mask_params(), counter, optimizer) #Then this newly rendered sketch is put into the loss function
        loss = sum(list(losses_dict.values()))
        print(loss)
        loss.backward()  # check that this step is working

        # # Directly accessing the activation mask parameters
        # activation_mask_params = renderer.activation_mask_parameters()
        #
        # # Checking if gradients are present
        # if activation_mask_params.grad is not None:
        #     print("Gradient for activation mask:", activation_mask_params.grad)
        #     print("Gradient Norm for activation mask:", activation_mask_params.grad.norm(2))
        # else:
        #     print("No gradient for activation mask")
        #
        # for param in renderer.activation_mask_parameters():
        #     assert param.requires_grad, "requires_grad is False for some activation_mask_parameters"

        optimizer.step_()
        if epoch % args.save_interval == 0:
            utils.plot_batch(inputs, sketches, f"{args.output_dir}/jpg_logs", counter,
                             use_wandb=args.use_wandb, title=f"iter{epoch}.jpg")
            renderer.save_png(
                f"{args.output_dir}/png_logs", f"png_iter{epoch}")
            # plotting()
        if epoch % args.eval_interval == 0:
            with torch.no_grad():
                losses_dict_eval = loss_func(sketches, inputs,
                                             renderer.get_activation_mask_params(), counter, optimizer, mode="eval")
                loss_eval = sum(list(losses_dict_eval.values()))
                configs_to_save["loss_eval"].append(loss_eval.item())
                for k in losses_dict_eval.keys():
                    if k not in configs_to_save.keys():
                        configs_to_save[k] = []
                    configs_to_save[k].append(losses_dict_eval[k].item())
                if args.clip_fc_loss_weight:
                    if losses_dict_eval["fc"].item() < best_fc_loss:
                        best_fc_loss = losses_dict_eval["fc"].item(
                        ) / args.clip_fc_loss_weight
                        best_iter_fc = epoch
                # print(
                #     f"eval iter[{epoch}/{args.num_iter}] loss[{loss.item()}] time[{time.time() - start}]")

                cur_delta = loss_eval.item() - best_loss
                if abs(cur_delta) > min_delta:
                    if cur_delta < 0:
                        best_loss = loss_eval.item()
                        best_iter = epoch
                        terminate = False
                        utils.plot_batch(
                            inputs, sketches, args.output_dir, counter, use_wandb=args.use_wandb, title="best_iter.jpg")
                        renderer.save_png(args.output_dir, "best_iter")

                # if args.use_wandb:
                #     wandb.run.summary["best_loss"] = best_loss
                #     wandb.run.summary["best_loss_fc"] = best_fc_loss
                #     wandb_dict = {"delta": cur_delta,
                #                   "loss_eval": loss_eval.item()}
                #     for k in losses_dict_eval.keys():
                #         wandb_dict[k + "_eval"] = losses_dict_eval[k].item()
                #     wandb.log(wandb_dict, step=counter)

                if abs(cur_delta) <= min_delta:
                    if terminate:
                        break
                    terminate = True

        if counter == 0 and args.attention_init:
            utils.plot_atten(renderer.get_attn(), renderer.get_thresh(), inputs, renderer.get_inds(),
                             args.use_wandb, "{}/{}.jpg".format(
                    args.output_dir, "attention_map"),
                             args.saliency_model, args.display_logs)

        # if args.use_wandb:
        #     wandb_dict = {"loss": loss.item(), "lr": optimizer.get_lr()}
        #     for k in losses_dict.keys():
        #         wandb_dict[k] = losses_dict[k].item()
        #     wandb.log(wandb_dict, step=counter)

        counter += 1

    renderer.save_png(args.output_dir, "final_png")
    path_png = os.path.join(args.output_dir, "best_iter.png")
    # utils.log_sketch_summary_final(
    #     path_png, args.use_wandb, args.device, best_iter, best_loss, "best total")



    return configs_to_save


if __name__ == "__main__":
    args = config.parse_arguments()

    final_config = vars(args)
    # try:
    configs_to_save = main(args)
    # except BaseException as err:
    #     print(f"Unexpected error occurred:\n {err}")
    #     print(traceback.format_exc())
    #     sys.exit(1)
    for k in configs_to_save.keys():
        final_config[k] = configs_to_save[k]
    np.save(f"{args.output_dir}/config.npy", final_config)

    final_config = np.load(f"{args.output_dir}/config.npy", allow_pickle=True).item()
    # Access the loss_eval list
    loss_eval = final_config['loss_eval']
    epochs = list(range(1, len(loss_eval) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_eval, label='Loss', marker='o', linestyle='-')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.output_dir}/loss.png")

    # if args.use_wandb:
    #     wandb.finish()

