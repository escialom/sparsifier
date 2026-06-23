# Phosphene Optimizer for Visual Prostheses

A deep learning framework for generating optimized phosphene representations of natural images for visual prosthesis research.

The project trains a convolutional neural network to learn phosphene placement maps that maximize the perceptual quality of electrically stimulated visual representations while controlling phosphene density. The framework integrates the DynaPhos simulator (van der Grinten et al., 2024), contour extraction methods, and CLIP-based perceptual objectives to generate and evaluate phosphene stimuli (CLIPASSO; Vinker et al., 2022).

---

## Overview

Visual prostheses restore limited vision by electrically stimulating the visual pathway, producing percepts known as *phosphenes*. Because the number of available electrodes is limited, selecting the most informative image regions is a critical challenge.

This repository implements a neural-network-based phosphene optimization pipeline that:

- Learns spatial phosphene placement maps from images
- Generates simulated phosphene vision using DynaPhos (van der Grinten et al., 2024)
- Supports contour-based and luminance-based baselines
- Uses perceptual losses (including CLIP-based objectives from Vinker et al., 2022)
- Produces matched phosphene stimuli for behavioral experiments
  
---

## Main Features

### Deep Learning Optimization
- CNN-based phosphene placement prediction
- PyTorch implementation
- Automatic training and validation pipeline
- Learning-rate warmup and cosine scheduling support

### Phosphene Simulation
- Integration with DynaPhos
- Probabilistic phosphene coordinate generation
- Adjustable phosphene density
- Reproducible simulations using fixed seeds

### Stimulus Generation
The framework can generate:

1. **DNN-Optimized Stimuli**
2. **Contour-Based Stimuli**
3. **Luminance-Based Stimuli**

Stimuli can be matched for the number of active electrodes to enable fair experimental comparisons.

---

## Repository Structure

```text
project/
│
├── config/
│   └── model_config.py
│
├── src/
│   ├── model.py
│   ├── StimGen.py
│   ├── ContourExtract.py
│   └── utils.py
│
├── dynaphos/
│   ├── simulator.py
│   ├── cortex_models.py
│   ├── image_processing.py
│   └── plotting.py
│
├── clipasso/
│   ├── models/
│   └── painterly_rendering.py
│
├── scripts/
│   ├── training_model.py
│   ├── stimulus_generation.py
│   ├── clip_semantic_consistency.py
│   ├── background_extraction.py
│   ├── get_contours.py
│   └── luminance_matching.py
│
├── data/
├── output/
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/escialom/object-sparsifier-prosthetic-vision.git
cd object-sparsifier-prosthetic-vision
```

### Create a Conda Environment

```bash
conda create -n phosphene python=3.11
conda activate phosphene
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training

The main training script is:

```bash
python scripts/training_model.py
```

Important training parameters can be modified in:

```text
config/model_config.py
```

Examples include:

- Learning rate
- Number of epochs
- Batch size
- Image size
- Phosphene density

---

## Generating Stimuli

After training a model, generate phosphene stimuli using:

```bash
python scripts/stimulus_generation.py
```

The script:

- Loads a trained checkpoint
- Generates optimized phosphene images
- Generates contour-based stimuli
- Generates luminance-based stimuli
- Matches all conditions for electrode count
- Saves the resulting stimuli

---

## Model Architecture

The optimization model consists of:

### Encoder
- Convolutional layers
- Max-pooling operations
- LeakyReLU activations

### Decoder
- Transposed convolutions
- Spatial upsampling
- Sigmoid output activation

The network predicts a phosphene placement map that is subsequently converted into a simulated phosphene image through the DynaPhos simulator (van der Grinten et al., 2024).

---

## Example Workflow

### 1. Prepare Dataset

```text
train_set/
├── class_1/
├── class_2/
└── ...

val_set/
├── class_1/
├── class_2/
└── ...
```

### 2. Preprocess the dataset

```bash
python scripts/RGBA_to_RGB.py
python scripts/background_extraction.py
python scripts/clip_semantic_consistency.py
```

### 3. Train the Model

```bash
python scripts/training_model.py
```

### 4. Generate Stimuli

```bash
python scripts/stimulus_generation.py
```

### 5. Postprocessing for psychophysical experiments
```bash
python scripts/luminance_matching.py
```

---

## Dependencies

Major libraries used in this project include:

- PyTorch
- TorchVision
- OpenCV
- NumPy
- SciPy
- Scikit-Image
- Scikit-Learn
- Matplotlib
- Plotly
- Weights & Biases (W&B)
- DynaPhos
- CLIP
- U²-Net

---

## Citation

If you use this code in academic work, please cite:

```bibtex
@software{phosphene_optimizer,
  title  = {Phosphene Optimizer for Visual Prostheses},
  author = {Your Name},
  year   = {2026},
  url    = {https://github.com/<username>/<repository>}
}
```

---

## License

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).

---

## Contact

For questions, bug reports, or collaborations, please open an issue or contact the repository maintainer.
