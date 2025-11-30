# Retinal Disease Classifier

A deep learning project for **multi-label classification of retinal diseases** from fundus images using **ConvNeXt-Tiny** architecture with transfer learning.

## Overview

This project implements an automated retinal disease detection system that can identify **43 different eye conditions** from retinal fundus photographs. The model uses a pre-trained ConvNeXt-Tiny backbone fine-tuned on the RFMiD (Retinal Fundus Multi-disease Image Dataset).

## Key Features

- **Multi-label Classification**: Detects multiple diseases in a single image
- **Transfer Learning**: Leverages ImageNet pre-trained ConvNeXt-Tiny weights
- **High Accuracy**: Achieves 98.5% overall accuracy on test set
- **Comprehensive Augmentation**: Rotation, flipping, color jittering, and more
- **Class Imbalance Analysis**: Detailed analysis of dataset challenges

## Model Performance

| Metric                   | Value  |
| ------------------------ | ------ |
| Test Accuracy            | 98.5%  |
| Test Loss                | 0.0494 |
| Learned Classes (F1 > 0) | 16/43  |
| Top Classes F1 Score     | 70-80% |

## Detected Diseases

The model is trained to detect 43 retinal conditions including:

- Diabetic Retinopathy (DR)
- Age-Related Macular Degeneration (ARMD)
- Macular Hole (MH)
- Diabetic Nephropathy (DN)
- Myopia (MYA)
- Branch Retinal Vein Occlusion (BRVO)
- And 37 more conditions...

## Tech Stack

- **Framework**: PyTorch
- **Model**: ConvNeXt-Tiny (ImageNet pretrained)
- **Loss Function**: BCEWithLogitsLoss
- **Optimizer**: AdamW with discriminative learning rates
- **Environment**: Google Colab (A100 GPU)

## Dataset

- **Source**: https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification
- **Training**: 1920 images
- **Validation**: 640 images
- **Test**: 640 images
- **Image Size**: 224x224 (resized)

## Quick Start

1. Clone the repository

```bash
git clone https://github.com/enesfreigthinus/retinal-disease-classifier.git
```
