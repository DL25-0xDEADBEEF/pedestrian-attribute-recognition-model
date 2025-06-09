# Pedestrian Attribute Recognition Model

> **Term Project for Deep Learning Course**  
> School of Computing, Gachon University  
> Spring 2025


A PyTorch implementation of pedestrian attribute recognition using Inception-based architecture with spatial transformation blocks for multi-attribute classification in surveillance scenarios.

## Overview

This project implements a deep learning model for pedestrian attribute recognition that can identify multiple attributes such as gender, age, clothing type, accessories, and more from pedestrian images. The model uses a modified BN-Inception architecture enhanced with spatial transformation blocks for improved feature extraction.

## Features

- **Multi-attribute Classification**: Recognizes multiple pedestrian attributes simultaneously
- **Spatial Transformation Blocks**: Enhanced feature extraction with attention mechanisms
- **Multiple Dataset Support**: Compatible with PA-100K, RAP, PETA, and custom datasets
- **Weighted BCE Loss**: Handles class imbalance with dataset-specific weight configurations
- **Deep Supervision**: Multiple prediction heads for improved training

## Supported Attributes

The model can recognize various pedestrian attributes including:

### PA-100K Dataset (26 attributes)
- Demographics: Female, Age groups (Less18, 18-60, Over60)
- Pose: Front, Side, Back
- Accessories: Hat, Glasses, HandBag, ShoulderBag, Backpack
- Clothing: ShortSleeve, LongSleeve, LongCoat, Trousers, Shorts, Skirt&Dress
- Patterns: UpperStride, UpperLogo, UpperPlaid, UpperSplice, LowerStripe, LowerPattern

### PETA Dataset (35 attributes)
- Age groups, clothing types, accessories, footwear, and more

### RAP Dataset (51 attributes)
- Comprehensive attribute set including body type, hair, clothing, and activities

## Requirements

```bash
torch>=1.7.0
torchvision>=0.8.0
numpy
PIL
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DL25-0xDEADBEEF/pedestrian-attribute-recognition.git
cd pedestrian-attribute-recognition
```

2. Install dependencies:
```bash
pip install torch torchvision numpy pillow
```

3. Download the pretrained BN-Inception weights:
```bash
# Place bn_inception-52deb4733.pth in the model/ directory
```

## Dataset Preparation

### Directory Structure
```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
    ├── train_list.txt
    ├── val_list.txt
    └── test_list.txt
```

### Annotation Format
Each line in the annotation files should follow this format:
```
image_path label1 label2 label3 ... labelN
```

Example:
```
person_001.jpg 1 0 1 0 1 0 0 1 1 0 ...
person_002.jpg 0 1 0 1 0 1 1 0 0 1 ...
```

Where each label is binary (0 or 1) corresponding to the absence or presence of each attribute.

## Usage

### Training

```bash
python main.py \
    --experiment peta \
    --approach inception_iccv \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --optimizer adam
```

### Evaluation

```bash
python main.py \
    --experiment peta \
    --approach inception_iccv \
    --evaluate \
    --resume path/to/checkpoint.pth.tar
```

### Parameters

- `--experiment`: Dataset name (peta, rap, pa100k, foottraffic)
- `--approach`: Model architecture (inception_iccv)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--optimizer`: Optimizer type (adam, sgd)
- `--resume`: Path to checkpoint for resuming training
- `--evaluate`: Run evaluation only

## Model Architecture

The model consists of:

1. **BN-Inception Backbone**: Modified Inception network for feature extraction
2. **Spatial Transformation Blocks**: Attention-based modules for attribute-specific feature learning
3. **Multi-scale Feature Fusion**: Combines features from different network depths
4. **Multiple Prediction Heads**: Separate predictions for different resolution features

### Key Components

- **ChannelAttn**: Channel attention mechanism for feature enhancement
- **SpatialTransformBlock**: Spatial transformation with learnable attention
- **Weighted BCE Loss**: Addresses class imbalance in attribute distribution

## Results

The model provides detailed evaluation metrics including:
- Per-attribute accuracy
- Mean accuracy (mA)
- Overall accuracy, precision, recall, and F1-score

## Customization

### Adding New Datasets

1. Update the `attr_nums` and `description` dictionaries in datasets.py
2. Add dataset-specific weights in the `Weighted_BCELoss` class
3. Modify the `Get_Dataset` function to handle your dataset

### Custom Attributes

To add new attributes:
1. Update the attribute descriptions in datasets.py
2. Adjust the number of classes in `attr_nums`
3. Update the loss weights if using weighted BCE loss

## Model Checkpoints

Trained models are saved with the following information:
- Model state dictionary
- Optimizer state
- Best accuracy achieved
- Training epoch

## Acknowledgments

- ICCV 2019: "Improving Pedestrian Attribute Recognition with Weakly-supervised Multi-scale Attribute-specific Localization"
- FootTraffic Dataset for Pedestrian Attribute Recognition
- BN-Inception implementation based on [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)
- Weighted BCE Loss inspired by multi-attribute learning research
- Spatial transformation blocks for improved attribute localization
