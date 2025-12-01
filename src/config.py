"""
Configuration module for Retinal Disease Classifier.
Contains all hyperparameters, paths, and constants.
"""

import os
import torch

# ============================================
# DEVICE CONFIGURATION
# ============================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================
# DATA PATHS (Modify according to your setup)
# ============================================
# For Google Colab
DATA_ROOT = '/content/dataset'

# Alternative: Local paths
# DATA_ROOT = './data'

TRAIN_DIR = os.path.join(DATA_ROOT, 'Training_Set', 'Training_Set')
VAL_DIR = os.path.join(DATA_ROOT, 'Evaluation_Set', 'Evaluation_Set')
TEST_DIR = os.path.join(DATA_ROOT, 'Test_Set', 'Test_Set')

TRAIN_IMG_DIR = os.path.join(TRAIN_DIR, 'Training')
VAL_IMG_DIR = os.path.join(VAL_DIR, 'Validation')
TEST_IMG_DIR = os.path.join(TEST_DIR, 'Test')

TRAIN_LABEL_PATH = os.path.join(TRAIN_DIR, 'RFMiD_Training_Labels.csv')
VAL_LABEL_PATH = os.path.join(VAL_DIR, 'RFMiD_Validation_Labels.csv')
TEST_LABEL_PATH = os.path.join(TEST_DIR, 'RFMiD_Testing_Labels.csv')

# ============================================
# MODEL CONFIGURATION
# ============================================
MODEL_NAME = 'convnext_tiny'
NUM_CLASSES = 43  # 45 total - 2 excluded (HR, ODPM)
PRETRAINED_WEIGHTS = 'IMAGENET1K_V1'

# Columns to exclude from labels
EXCLUDE_COLUMNS = ['ID', 'Disease_Risk', 'HR', 'ODPM']

# ============================================
# TRAINING HYPERPARAMETERS
# ============================================
BATCH_SIZE = 64
NUM_WORKERS = os.cpu_count() or 4
PIN_MEMORY = True

# Learning Rate
LEARNING_RATE = 2e-3  # Found via LR Finder
LR_FEATURE_EXTRACTOR = LEARNING_RATE / 10  # Discriminative fine-tuning
LR_CLASSIFIER = LEARNING_RATE

# Training Settings
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0

# LR Finder Settings
LR_FINDER_START_LR = 1e-7
LR_FINDER_END_LR = 10
LR_FINDER_NUM_ITER = 100

# ============================================
# IMAGE PREPROCESSING
# ============================================
IMAGE_SIZE = 224
RESIZE_SIZE = 256  # For advanced augmentation

# ImageNet pretrained model normalization values
PRETRAINED_MEANS = [0.485, 0.456, 0.406]
PRETRAINED_STDS = [0.229, 0.224, 0.225]

# ============================================
# DATA AUGMENTATION SETTINGS
# ============================================
# Basic augmentation
ROTATION_DEGREES = 180
HORIZONTAL_FLIP_PROB = 0.5
VERTICAL_FLIP_PROB = 0.5
SHARPNESS_FACTOR = 2
SHARPNESS_PROB = 0.8

# Advanced augmentation
COLOR_JITTER = {
    'brightness': 0.3,
    'contrast': 0.3,
    'saturation': 0.3,
    'hue': 0.1
}

RANDOM_AFFINE = {
    'degrees': 180,
    'translate': (0.1, 0.1),
    'scale': (0.9, 1.1),
    'shear': 10
}

GAUSSIAN_BLUR_KERNEL = 3
GAUSSIAN_BLUR_SIGMA = (0.1, 2.0)

RANDOM_ERASING = {
    'p': 0.2,
    'scale': (0.02, 0.1),
    'ratio': (0.3, 3.3)
}

# ============================================
# LOSS FUNCTION SETTINGS
# ============================================
# Focal Loss
FOCAL_LOSS_GAMMA = 2.0

# ============================================
# OUTPUT PATHS
# ============================================
OUTPUT_DIR = './outputs'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

# Checkpoint naming
CHECKPOINT_TEMPLATE = 'convnext_cp_{epoch}.pth'
BEST_MODEL_TEMPLATE = 'convnext_best.pth'

# ============================================
# LOGGING
# ============================================
RESULTS_JSON = 'training_results.json'
METRICS_CSV = 'class_metrics.csv'

# ============================================
# HELPER FUNCTIONS
# ============================================
def create_output_dirs():
    """Create output directories if they don't exist."""
    for dir_path in [OUTPUT_DIR, CHECKPOINT_DIR, RESULTS_DIR, PLOTS_DIR]:
        os.makedirs(dir_path, exist_ok=True)

def get_config_dict():
    """Return configuration as dictionary for logging."""
    return {
        'device': DEVICE,
        'model_name': MODEL_NAME,
        'num_classes': NUM_CLASSES,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'weight_decay': WEIGHT_DECAY,
        'image_size': IMAGE_SIZE,
        'focal_loss_gamma': FOCAL_LOSS_GAMMA,
    }

def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    for key, value in get_config_dict().items():
        print(f"  {key}: {value}")
    print("=" * 60)
