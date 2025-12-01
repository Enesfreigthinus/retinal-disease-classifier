# Retinal Disease Classifier

A deep learning project for **multi-label classification of retinal diseases** from fundus images using **ConvNeXt-Tiny** architecture with transfer learning.

## 📁 Project Structure

```
Retinal Disease Classifier/
├── src/                          # Modular source code
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Configuration and hyperparameters
│   ├── dataset.py                # Dataset classes and data loading
│   ├── transforms.py             # Data augmentation transforms
│   ├── model.py                  # Model architecture definitions
│   ├── losses.py                 # Loss functions (Focal, BCE, etc.)
│   ├── metrics.py                # Evaluation metrics
│   ├── trainer.py                # Training and evaluation loops
│   ├── visualization.py          # Plotting and visualization
│   └── utils.py                  # Utility functions (LR Finder, etc.)
├── train.py                      # Main training script
├── inference.py                  # Inference/prediction script
├── requirements.txt              # Python dependencies
├── README.md                     # Documentation
└── retinalDiseaseClassifier_latest.ipynb  # Original notebook
```

## Overview

This project implements an automated retinal disease detection system that can identify **43 different eye conditions** from retinal fundus photographs. The model uses a pre-trained ConvNeXt-Tiny backbone fine-tuned on the RFMiD (Retinal Fundus Multi-disease Image Dataset).

## Key Features

- **Multi-label Classification**: Detects multiple diseases in a single image
- **Transfer Learning**: Leverages ImageNet pre-trained ConvNeXt-Tiny weights
- **Class Imbalance Handling**: Focal Loss, Weighted BCE, Weighted Random Sampler
- **Advanced Augmentation**: ColorJitter, RandomAffine, GaussianBlur, RandomErasing
- **Modular Architecture**: Clean, reusable code structure
- **Comprehensive Metrics**: Precision, Recall, F1, ROC curves, Confusion matrices

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
- **Loss Functions**: Focal Loss, Weighted BCE, Asymmetric Loss
- **Optimizer**: AdamW with discriminative learning rates
- **Scheduler**: Cosine Annealing LR
- **Environment**: Google Colab (GPU) / Local

## Dataset

- **Source**: https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification
- **Training**: 1920 images
- **Validation**: 640 images
- **Test**: 640 images
- **Image Size**: 224x224 (resized)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/enesfreigthinus/retinal-disease-classifier.git
cd retinal-disease-classifier

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Basic training
python train.py

# With custom parameters
python train.py --epochs 30 --batch-size 64 --lr 2e-3 --loss focal

# Run LR finder first
python train.py --find-lr --epochs 30
```

### Inference

```bash
# Single image prediction
python inference.py --image ./test_image.png --model ./outputs/checkpoints/best_model.pth

# Batch prediction on folder
python inference.py --folder ./test_images/ --output predictions.csv
```

## 📊 Usage Examples

### Using the Modular Code

```python
from src import config, dataset, transforms, model, losses, trainer

# Setup transforms
train_transform = transforms.get_advanced_train_transform()
test_transform = transforms.get_test_transform()

# Create datasets
data = dataset.setup_datasets(train_transform, test_transform)

# Create model
net = model.create_convnext_model()

# Create loss function with class weighting
loss_fn = losses.get_loss_function(
    loss_type='focal',
    pos_weight=data['pos_weight']
)

# Train
train_manager = trainer.Trainer(
    model=net,
    train_loader=data['train_loader'],
    val_loader=data['val_loader'],
    loss_fn=loss_fn,
    optimizer=model.get_optimizer(net)
)
train_manager.train(num_epochs=30)
```

### Custom Inference

```python
from inference import RetinalDiseasePredictor

predictor = RetinalDiseasePredictor(model_path='./best_model.pth')
result = predictor.predict_single('./retinal_image.png')

print(f"Detected diseases: {result['predicted_diseases']}")
print(f"Probabilities: {result['probabilities']}")
```
