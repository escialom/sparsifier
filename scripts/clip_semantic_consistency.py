"""
Perform automated semantic quality control on a preprocessed image dataset
using a pre-trained CLIP (RN101) model.

After background extraction and resizing, this script filters out images
whose semantic content is not reliably preserved. Imperfect segmentation
(e.g., from U2-Net-based background removal) may introduce artifacts or
remove important object parts. To mitigate this, a zero-shot classification
consistency check is applied using CLIP.

Pipeline
--------
1. Load a pre-trained CLIP model (RN101 variant) and its preprocessing pipeline.
2. Define a fixed set of class labels corresponding to dataset categories.
3. Encode text prompts of the form "a picture of a {class}" to obtain text features.
4. Iterate over all images in the dataset:
    a. Preprocess each image using CLIP’s preprocessing pipeline.
    b. Encode the image into CLIP feature space.
    c. L2-normalize both image and text features.
    d. Compute cosine similarities via scaled dot product.
    e. Apply softmax to obtain class probabilities.
5. Remove an image if:
    - The maximum predicted class probability is below 0.5 (low semantic confidence), or
    - The predicted class label does not match the ground-truth label.

This procedure ensures that only images with semantically consistent and
recognizable foreground objects are retained, improving dataset quality
for subsequent CLIP-guided phosphene optimization.

Notes
-----
- The classification is performed in a zero-shot setting using CLIP.
- Similarity scores are computed as:
      logits = 100 * (image_features @ text_features.T)
- Feature vectors are L2-normalized prior to similarity computation.
- Images failing the quality criteria are permanently deleted from disk.

Parameters
----------
args : argparse.Namespace
    Configuration object containing runtime arguments. Must include:
    - device: computation device (e.g., "cuda" or "cpu")

Variables
---------
device : torch.device
    Device used for model inference.
model : torch.nn.Module
    Pre-trained CLIP RN101 model.
clip_preprocess : torchvision.transforms.Compose
    CLIP preprocessing pipeline applied to each image.
classnames : list of str
    List of dataset class labels used for zero-shot classification.
text_features : torch.Tensor
    Encoded and normalized text embeddings for all class prompts.
dataloader : torch.utils.data.DataLoader
    DataLoader over the preprocessed dataset (batch_size=1).
n_removed : int
    Counter tracking the number of removed images.

Side Effects
------------
- Deletes image files from disk that fail the semantic consistency check.
- Prints information about removed files.
"""

import os
import torch
import clipasso.CLIP_.clip as clip
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
from config import model_config

args = model_config.parse_arguments()
device = torch.device(args.device)
model, clip_preprocess = clip.load("RN101", device, jit=False)
img_size = clip_preprocess.transforms[1].size
model.eval()
classnames = ["banana", "boot", "bowl", "cup", "glasses", "hat", "lamp", "pan", "sewing_machine", "shovel"]
text = clip.tokenize([f"a picture of a {c}" for c in classnames]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text)

n_removed = 0

input_folder = ImageFolder("./data_preprocessed_white_bg/val_set")
dataloader = DataLoader(input_folder, batch_size=1)

for file_path, label_idx in dataloader.dataset.samples:
    img_path = Path(file_path)
    with Image.open(file_path) as im:
        img = clip_preprocess(im).unsqueeze(0).to(device)
    with torch.no_grad():
        # Encode image into CLIP feature space
        image_features = model.encode_image(img)
        # Normalize features to unit length (cosine similarity)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = text_features.detach()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # Compute similarity (scaled cosine similarity)
        logits_per_image = (100.0 * image_features @ text_features.T)
        # Convert to probabilities
        probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()
        prob_dict = dict(zip(classnames, probs[0]))
        max_prob = max(prob_dict.values())
        model_guess = max(prob_dict, key=prob_dict.get)
        current_class = dataloader.dataset.classes[label_idx]
        # Probability should be at least 0.5 to ensure acceptable semantic confidence
        # The model should get the correct class (ground truth)
        if max_prob < 0.5 or model_guess != current_class:
            print(f"About to remove {file_path} because of {max_prob} or {model_guess}.")
            os.remove(file_path)
            n_removed += 1
            print(f"{file_path} removed, number of file removed = {n_removed}")