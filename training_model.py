import torch
import config
from torch.utils.data import DataLoader
import clipasso
from model import PhospheneOptimizer
from tqdm.auto import tqdm
from clipasso.models.loss import Loss


# Load model and phosphene simulator parameters
model_params = config.model_config.parse_arguments()

import os
from pathlib import Path
import numpy as np
from torchvision import transforms
from clipasso import sketch_utils as utils


def train(model_params):

    model = PhospheneOptimizer()
    model().to(model_params.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params.lr)
    loss_func = Loss(model_params)
    train_loader = DataLoader(model_params.train_set, batch_size=model_params.batch_size, shuffle=True)

    train_loss = 0
    iter = 0
    epoch_loss = []
    if model_params.display:
        epoch_range = range(model_params.num_iter)
    else:
        epoch_range = tqdm(range(model_params.num_iter))
    for epoch in epoch_range:
        for batches, (inputs, targets) in enumerate(train_loader):
            # Data loading and augmentation performed by clipasso
            input_img, mask = clipasso.painterly_rendering.get_target(model_params, inputs)
            input_img = input_img.to(model_params.device)
            optimizer.zero_grad()
            output_img = model(input_img)
            #output_img[output_img == 1.] = 0.  # Make image background black
            losses_dict = loss_func(output_img, input_img, model.parameters(), iter, optimizer)
            loss = sum(losses_dict.values())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            iter += 1
        # Save loss of each epoch
        epoch_loss.append({
            "epoch": epoch + 1,
            "train_loss": train_loss
        })
    return epoch_loss




# def main_old(model_params):
#
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])
#
#     # Define the paths for training and validation datasets
#     train_dir = Path(os.path.join(model_params.data_dir, 'train'))
#     val_dir = Path(os.path.join(model_params.data_dir, 'val'))
#
#     train_dataset = ImagePathDataset(root_dir=train_dir, transform=transform)
#     val_dataset = ImagePathDataset(root_dir=val_dir, transform=transform)
#
#     train_loader = DataLoader(train_dataset, batch_size=model_params.batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=model_params.batch_size, shuffle=False)
#
#     # visualize_batch(train_loader)
#
#     loss_func = Loss(model_params)
#     phosphene_model = Phosphene_model(model_params,
#                                       electrode_grid=model_params.electrode_grid,
#                                       imsize=224,
#                                       device='cpu').to(model_params.device)
#     optimizer = torch.optim.Adam(phosphene_model.parameters(), lr=model_params.lr)
#
#     print(count_trainable_parameters(phosphene_model))  # STN parameters
#
#     best_loss = float('inf')
#     epoch_data = []
#     counter = 0
#     configs_to_save = {"loss_eval": []}
#     # best_loss = 100
#     # best_fc_loss = 100
#     # best_iter, best_iter_fc = 0, 0
#     min_delta = 1e-5
#     terminate = False
#
#     # Select an image to track over epochs
#     sample_index = 12
#     sample_image, sample_label, sample_image_id, sample_image_path = train_dataset[sample_index]
#     sample_image = sample_image.unsqueeze(0).to(model_params.device)
#
#     if model_params.display:
#         epoch_range = range(model_params.num_iter)
#     else:
#         epoch_range = tqdm(range(model_params.num_iter))
#
#     for epoch in epoch_range:
#         phosphene_model.train()
#         train_loss = 0.0
#         for target_images, labels, image_ids, paths in train_loader:
#             for i, image_path in enumerate(paths):
#                 target_im, mask = get_target_and_mask(model_params, target_image_path=image_path)
#                 target_im = target_im.to(model_params.device)
#                 optimizer.zero_grad()
#                 optimized_im = phosphene_model(target_im, model_params)  # Process the entire batch
#                 optimized_im = optimized_im.to(model_params.device)
#
#                 # Apply custom operation on target images
#                 target_im[target_im == 1.] = 0.  # Make image background black
#
#                 losses_dict = loss_func(optimized_im, target_im, phosphene_model.parameters(), counter, optimizer)
#                 loss = sum(losses_dict.values())
#                 loss.backward()
#                 optimizer.step()
#                 train_loss += loss.item()
#                 counter += 1
#
#         train_loss /= len(train_loader)
#
#         phosphene_model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for images, labels, image_ids, paths in val_loader:
#                 for i, image_path in enumerate(paths):
#                     target_im, mask = get_target_and_mask(model_params, target_image_path=image_path)
#                     target_im = target_im.to(model_params.device)
#
#                     optimized_im = phosphene_model(target_im, model_params)
#                     optimized_im = optimized_im.to(model_params.device)
#
#                     target_im[target_im == 1.] = 0.  # Make image background black
#
#                     losses_dict = loss_func(optimized_im, target_im, phosphene_model.parameters(), counter, optimizer,
#                                             mode="eval")
#                     loss = sum(losses_dict.values())
#                     val_loss += loss.item()
#
#         val_loss /= len(val_loader)
#
#         print(f"Epoch {epoch + 1}/{model_params.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
#
#         epoch_data.append({
#             "epoch": epoch + 1,
#             "train_loss": train_loss,
#             "val_loss": val_loss
#         })
#
#         if val_loss < best_loss:
#             best_loss = val_loss
#             torch.save(phosphene_model.state_dict(), os.path.join(model_params.output_dir, "best_model.pth"))
#
#         # Save the current state of the sample image every 'save_interval' epochs during training
#         if (epoch + 1) % model_params.save_interval == 0:
#             phosphene_model.eval()
#             with torch.no_grad():
#                 sample_optimized_im = phosphene_model(sample_image, model_params)
#             utils.plot_batch(sample_image, sample_optimized_im, f"{model_params.output_dir}/jpg_logs", counter,
#                              title=f"train_iter{epoch + 1}.jpg")
#             phosphene_model.save_png(f"{model_params.output_dir}/png_logs", f"train_png_iter{epoch + 1}", sample_optimized_im)
#
#         # Save the current state of the sample image every 'save_interval' epochs during validation
#         if (epoch + 1) % model_params.save_interval == 0:
#             with torch.no_grad():
#                 sample_optimized_im = phosphene_model(sample_image, model_params)
#             utils.plot_batch(sample_image, sample_optimized_im, f"{model_params.output_dir}/jpg_logs", counter,
#                              title=f"val_iter{epoch + 1}.jpg")
#             phosphene_model.save_png(f"{model_params.output_dir}/png_logs", f"val_png_iter{epoch + 1}", sample_optimized_im)
#
#         save_epoch_data(model_params.output_dir, epoch_data)
#
#         # Save the final state of the sample image after the last epoch
#     phosphene_model.eval()
#     with torch.no_grad():
#         final_sample_optimized_im = phosphene_model(sample_image, model_params)
#     utils.plot_batch(sample_image, final_sample_optimized_im, f"{model_params.output_dir}/jpg_logs", counter,
#                      title="final_sample.jpg")
#     phosphene_model.save_png(f"{model_params.output_dir}/png_logs", "final_sample_png", final_sample_optimized_im)
#
#     # Save the final state of the last processed image
#     phosphene_model.save_png(model_params.output_dir, "final_png", optimized_im)
#     return epoch_data
#
#
# if __name__ == "__main__":
#     model_params = config.model_config.parse_arguments()
#     final_config = vars(model_params)
#     configs_to_save = main(model_params)
#     for k in configs_to_save.keys():
#         final_config[k] = configs_to_save[k]
#     np.save(f"{model_params.output_dir}/config.npy", final_config)
