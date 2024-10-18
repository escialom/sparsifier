import utils

utils.plot_losses("./output", "training_data_checkpoints.pth", "validation_data_checkpoints.pth")
utils.plot_stim_properties("./output", "data_tracked_validation_img.pth")
utils.val_img_properties("./output", "data_val_imgs_after_training.pth")