from models.loss import Loss
from simplified_painter import Phosphene_model
import config
import sketch_utils as utils
import torch
from simplified_painter import get_target_and_mask
from tqdm.auto import tqdm
import time
import os
import numpy as np
from pathlib import Path


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# TODO check input in the loss (check phosphene im, if dynaphos gives something strange aka extreme values,
#  and look at dynaphos demo
def main(args):

    loss_func = Loss(args)
    my_real_super_secret_vip_loss_func = torch.nn.MSELoss()
    phosphene_model = Phosphene_model(args,
                                      num_phosphenes=args.num_phosphenes,
                                      imsize=224,
                                      device='cpu').to(args.device)
    optimizer = torch.optim.Adam(phosphene_model.parameters(), lr=args.lr)


    print(count_trainable_parameters(phosphene_model)) #79122794 #178969707 # 27692394, these are the stn params

    counter = 0
    configs_to_save = {"loss_eval": []}
    best_loss = 100
    best_fc_loss = 100
    best_iter, best_iter_fc = 0, 0
    min_delta = 1e-5
    terminate = False

    abs_path = Path(os.path.abspath(os.getcwd()))
    # path_to_image_directory = ...
    # image_paths = os.listdir(path_to_image_directory)
    image_paths = [Path(f"{abs_path}/target_images/camel.png"), ]

    if args.display:
        epoch_range = range(args.num_iter)
    else:
        epoch_range = tqdm(range(args.num_iter))

    for epoch in epoch_range:
        for image_path in image_paths:
            with torch.autograd.detect_anomaly():
                target_im, mask = get_target_and_mask(args, target_image_path=image_path)
                target_im[target_im == 1.] = 0.
                # target_im.requires_grad = True

                # Check target_im for NaNs or infinite values right after it's prepared
                assert not torch.isnan(target_im).any(), "target_im contains NaN values"
                assert not torch.isinf(target_im).any(), "target_im contains infinite values"

                if not args.display:
                    epoch_range.refresh()

                optimizer.zero_grad()

                phosphene_im = phosphene_model(target_im, args)
                phosphene_im.to(args.device)

                print(torch.isnan(phosphene_im).any())
                print(torch.isinf(phosphene_im).any())

                # Check phosphene_im for NaNs or infinite values right after it's generated
                assert not torch.isnan(phosphene_im).any(), "phosphene_im contains NaN values"
                assert not torch.isinf(phosphene_im).any(), "phosphene_im contains infinite values"

                losses_dict = loss_func(phosphene_im, target_im.detach(
                ), phosphene_model.parameters(), counter,
                                        optimizer) #TODO why does phosphene_im become nan?

                # print("Before backpass: the parameters and sizes")
                #
                # for name, param in phosphene_model.named_parameters():
                #     print(name, param.size())
                #
                # print("Before backpass: the parameters and values")
                # for name, param in phosphene_model.named_parameters():
                #     print(name, param)  # This prints the parameter tensors directly
                #     assert not torch.isnan(param).any(), ['NaN detected in ' + name]



                #loss = sum(list(losses_dict.values()))
                loss = my_real_super_secret_vip_loss_func(phosphene_im, target_im.detach())
                print(loss)
                loss.backward()  # check that this step is working

                def contains_nan(tensor):
                    return torch.isnan(tensor).any()

                # After loss.backward()
                if any(contains_nan(param.grad) for param in phosphene_model.parameters()):
                    print("NaNs detected in gradients.")

                # torch.nn.utils.clip_grad_norm_(phosphene_model.parameters(), max_norm=1.0) #Gradient clipping gives only nan values

                optimizer.step()

                # print("After backpass: the parameters and sizes")
                # for name, param in phosphene_model.named_parameters():
                #     print(name, param)  # This prints the parameter tensors directly
                    # assert not torch.isnan(param).any(), ['NaN detected in ' + name]

                if epoch % args.save_interval == 0:
                    utils.plot_batch(target_im, phosphene_im, f"{args.output_dir}/jpg_logs", counter,
                                     use_wandb=args.use_wandb, title=f"iter{epoch}.jpg")
                    phosphene_model.save_png(
                        f"{args.output_dir}/png_logs", f"png_iter{epoch}", phosphene_im)

                if epoch % args.eval_interval == 0:
                    with torch.no_grad():
                        losses_dict_eval = loss_func(phosphene_im, target_im,
                                                     phosphene_model.parameters(), counter, optimizer, mode="eval")
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

                        cur_delta = loss_eval.item() - best_loss
                        if abs(cur_delta) > min_delta:
                            if cur_delta < 0:
                                best_loss = loss_eval.item()
                                best_iter = epoch
                                terminate = False
                                utils.plot_batch(
                                    target_im, phosphene_im, args.output_dir, counter, use_wandb=args.use_wandb,
                                    title="best_iter.jpg")
                                phosphene_model.save_png(args.output_dir, "best_iter", phosphene_im)

                        if abs(cur_delta) <= min_delta:
                            if terminate:
                                break
                            terminate = True

                # if counter == 0 and args.attention_init:
                #     utils.plot_atten(attention_map, clip_saliency_map, target_im, phosphene_placement_map,
                #                      args.use_wandb, "{}/{}.jpg".format(
                #             args.output_dir, "attention_map"),
                #                      args.saliency_model, args.display_logs)

        counter += 1

    phosphene_model.save_png(args.output_dir, "final_png")
    path_png = os.path.join(args.output_dir, "best_iter.png")

    # utils.log_sketch_summary_final(
    #     path_png, args.use_wandb, args.device, best_iter, best_loss, "best total")


    return configs_to_save


if __name__ == "__main__":
    args = config.parse_arguments()

    final_config = vars(args)

    configs_to_save = main(args)

    for k in configs_to_save.keys():
        final_config[k] = configs_to_save[k]
    np.save(f"{args.output_dir}/config.npy", final_config)

