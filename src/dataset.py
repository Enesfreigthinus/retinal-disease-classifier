"""
Dataset module for Retinal Disease Classifier.
Contains dataset classes and data loading utilities.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

from . import config


def path2id(img_path: str) -> int:
    """Extract image ID from file path."""
    return int(os.path.splitext(os.path.basename(img_path))[0])


class RetinaDataset(Dataset):
    """
    PyTorch Dataset for Retinal Disease Classification.
    
    Handles multi-label classification where each retinal image
    can have multiple disease labels.
    
    Args:
        img_paths: List of image file paths
        label_csv_path: Path to CSV file containing labels
        transform: Optional torchvision transforms to apply
        exclude_columns: Columns to exclude from labels (default from config)
    """
    
    def __init__(
        self, 
        img_paths: list, 
        label_csv_path: str, 
        transform=None,
        exclude_columns: list = None
    ):
        self.img_paths = sorted(img_paths)
        self.transform = transform
        
        # Load and preprocess labels
        if exclude_columns is None:
            exclude_columns = config.EXCLUDE_COLUMNS
        
        self.label_df = pd.read_csv(label_csv_path)
        self.label_columns = [
            col for col in self.label_df.columns 
            if col not in exclude_columns
        ]
        self.labels = self.label_df[self.label_columns]
    
    def __len__(self) -> int:
        return len(self.img_paths)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            tuple: (image, label) where image is tensor and label is float tensor
        """
        img_path = self.img_paths[idx]
        img_id = path2id(img_path)
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Get label (ID - 1 because IDs are 1-indexed)
        label = np.array(self.labels.iloc[img_id - 1])
        
        if self.transform is not None:
            img = self.transform(img)
            label = torch.from_numpy(label).float()
        else:
            img = np.array(img)
            
        return img, label
    
    def get_class_names(self) -> list:
        """Return list of class/disease names."""
        return self.label_columns
    
    def get_label_df(self) -> pd.DataFrame:
        """Return the labels DataFrame."""
        return self.label_df


def get_image_paths(data_dir: str, pattern: str = '*.png') -> list:
    """
    Get all image paths from a directory.
    
    Args:
        data_dir: Directory containing images
        pattern: Glob pattern for image files
        
    Returns:
        List of image file paths
    """
    return glob.glob(os.path.join(data_dir, pattern))


def load_label_dataframe(label_path: str) -> pd.DataFrame:
    """Load label CSV file as DataFrame."""
    return pd.read_csv(label_path)


def calculate_class_frequencies(label_df: pd.DataFrame, exclude_columns: list = None) -> pd.Series:
    """
    Calculate frequency of each class in the dataset.
    
    Args:
        label_df: DataFrame containing labels
        exclude_columns: Columns to exclude
        
    Returns:
        Series with class frequencies
    """
    if exclude_columns is None:
        exclude_columns = config.EXCLUDE_COLUMNS
        
    label_cols = [col for col in label_df.columns if col not in exclude_columns]
    return label_df[label_cols].sum()


def calculate_pos_weight(label_df: pd.DataFrame, exclude_columns: list = None) -> torch.Tensor:
    """
    Calculate positive weights for class imbalance handling.
    
    pos_weight = negative_samples / positive_samples
    
    Args:
        label_df: DataFrame containing labels
        exclude_columns: Columns to exclude
        
    Returns:
        Tensor of positive weights for each class
    """
    if exclude_columns is None:
        exclude_columns = config.EXCLUDE_COLUMNS
        
    label_cols = [col for col in label_df.columns if col not in exclude_columns]
    labels = label_df[label_cols]
    
    pos_counts = labels.sum()
    neg_counts = len(labels) - pos_counts
    
    # Add small epsilon to avoid division by zero
    pos_weight = neg_counts / (pos_counts + 1)
    
    return torch.tensor(pos_weight.values, dtype=torch.float32)


def calculate_sample_weights(label_df: pd.DataFrame, exclude_columns: list = None) -> torch.Tensor:
    """
    Calculate sample weights for weighted random sampling.
    
    Samples with rare classes get higher weights for oversampling.
    
    Args:
        label_df: DataFrame containing labels
        exclude_columns: Columns to exclude
        
    Returns:
        Tensor of sample weights
    """
    if exclude_columns is None:
        exclude_columns = config.EXCLUDE_COLUMNS
        
    label_cols = [col for col in label_df.columns if col not in exclude_columns]
    labels = label_df[label_cols].values
    
    pos_counts = labels.sum(axis=0)
    
    # Class weights: inverse frequency
    class_weights = 1.0 / (pos_counts + 1)
    class_weights = class_weights / class_weights.sum()  # Normalize
    
    # Sample weights: sum of class weights for each sample
    sample_weights = (labels * class_weights).sum(axis=1)
    
    # Minimum weight for samples without any disease
    min_weight = sample_weights[sample_weights > 0].min()
    sample_weights = np.maximum(sample_weights, min_weight)
    
    return torch.tensor(sample_weights, dtype=torch.float32)


def create_weighted_sampler(sample_weights: torch.Tensor, replacement: bool = True) -> WeightedRandomSampler:
    """
    Create a weighted random sampler for handling class imbalance.
    
    Args:
        sample_weights: Weights for each sample
        replacement: Whether to sample with replacement
        
    Returns:
        WeightedRandomSampler instance
    """
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=replacement
    )


def create_data_loaders(
    train_data: Dataset,
    val_data: Dataset,
    test_data: Dataset,
    batch_size: int = None,
    num_workers: int = None,
    sampler=None
) -> tuple:
    """
    Create train, validation, and test data loaders.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        test_data: Test dataset
        batch_size: Batch size (default from config)
        num_workers: Number of workers (default from config)
        sampler: Optional sampler for training data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_workers is None:
        num_workers = config.NUM_WORKERS
    
    # Training loader
    train_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': config.PIN_MEMORY
    }
    
    if sampler is not None:
        train_kwargs['sampler'] = sampler
    else:
        train_kwargs['shuffle'] = True
        
    train_loader = DataLoader(train_data, **train_kwargs)
    
    # Validation loader
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY
    )
    
    # Test loader
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader, test_loader


def setup_datasets(train_transform, test_transform):
    """
    Complete dataset setup with default paths.
    
    Args:
        train_transform: Transform for training data
        test_transform: Transform for validation/test data
        
    Returns:
        Dictionary containing datasets, loaders, and metadata
    """
    # Get image paths
    train_img_paths = get_image_paths(config.TRAIN_IMG_DIR)
    val_img_paths = get_image_paths(config.VAL_IMG_DIR)
    test_img_paths = get_image_paths(config.TEST_IMG_DIR)
    
    # Create datasets
    train_data = RetinaDataset(train_img_paths, config.TRAIN_LABEL_PATH, train_transform)
    val_data = RetinaDataset(val_img_paths, config.VAL_LABEL_PATH, test_transform)
    test_data = RetinaDataset(test_img_paths, config.TEST_LABEL_PATH, test_transform)
    
    # Load label dataframe for computing weights
    train_label_df = load_label_dataframe(config.TRAIN_LABEL_PATH)
    
    # Calculate weights
    pos_weight = calculate_pos_weight(train_label_df)
    sample_weights = calculate_sample_weights(train_label_df)
    
    # Create sampler
    weighted_sampler = create_weighted_sampler(sample_weights)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data,
        sampler=weighted_sampler
    )
    
    # Get class names
    class_names = train_data.get_class_names()
    
    return {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_label_df': train_label_df,
        'pos_weight': pos_weight,
        'sample_weights': sample_weights,
        'class_names': class_names,
        'num_train': len(train_data),
        'num_val': len(val_data),
        'num_test': len(test_data)
    }
