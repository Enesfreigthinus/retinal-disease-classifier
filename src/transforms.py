"""
Transforms module for Retinal Disease Classifier.
Contains data augmentation and preprocessing transforms.
"""

import torch
import torchvision.transforms.v2 as transforms

from . import config


def get_basic_train_transform() -> transforms.Compose:
    """
    Get basic training transforms with standard augmentations.
    
    Includes:
        - Resize
        - Random sharpness
        - Random rotation
        - Random horizontal/vertical flip
        - Center crop
        - Normalization
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomAdjustSharpness(
            config.SHARPNESS_FACTOR, 
            config.SHARPNESS_PROB
        ),
        transforms.RandomRotation(config.ROTATION_DEGREES),
        transforms.RandomHorizontalFlip(config.HORIZONTAL_FLIP_PROB),
        transforms.RandomVerticalFlip(config.VERTICAL_FLIP_PROB),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=config.PRETRAINED_MEANS, 
            std=config.PRETRAINED_STDS
        )
    ])


def get_advanced_train_transform() -> transforms.Compose:
    """
    Get advanced training transforms for better generalization.
    
    Includes all basic transforms plus:
        - ColorJitter
        - RandomAffine
        - GaussianBlur
        - RandomErasing (Cutout-like)
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(config.RESIZE_SIZE),
        
        # Color augmentations
        transforms.ColorJitter(
            brightness=config.COLOR_JITTER['brightness'],
            contrast=config.COLOR_JITTER['contrast'],
            saturation=config.COLOR_JITTER['saturation'],
            hue=config.COLOR_JITTER['hue']
        ),
        
        # Geometric augmentations
        transforms.RandomAffine(
            degrees=config.RANDOM_AFFINE['degrees'],
            translate=config.RANDOM_AFFINE['translate'],
            scale=config.RANDOM_AFFINE['scale'],
            shear=config.RANDOM_AFFINE['shear']
        ),
        transforms.RandomHorizontalFlip(config.HORIZONTAL_FLIP_PROB),
        transforms.RandomVerticalFlip(config.VERTICAL_FLIP_PROB),
        
        # Sharpness and blur
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.GaussianBlur(
            kernel_size=config.GAUSSIAN_BLUR_KERNEL,
            sigma=config.GAUSSIAN_BLUR_SIGMA
        ),
        
        # Final processing
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=config.PRETRAINED_MEANS, 
            std=config.PRETRAINED_STDS
        ),
        
        # Random erasing (Cutout-like)
        transforms.RandomErasing(
            p=config.RANDOM_ERASING['p'],
            scale=config.RANDOM_ERASING['scale'],
            ratio=config.RANDOM_ERASING['ratio']
        ),
    ])


def get_test_transform() -> transforms.Compose:
    """
    Get test/validation transforms (no augmentation).
    
    Includes:
        - Resize
        - Center crop
        - Normalization
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=config.PRETRAINED_MEANS, 
            std=config.PRETRAINED_STDS
        )
    ])


def get_inference_transform() -> transforms.Compose:
    """
    Get inference transforms (same as test).
    
    Returns:
        Composed transforms
    """
    return get_test_transform()


def denormalize_image(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize an image tensor for visualization.
    
    Args:
        img_tensor: Normalized image tensor (C, H, W)
        
    Returns:
        Denormalized image tensor
    """
    mean = torch.tensor(config.PRETRAINED_MEANS).view(3, 1, 1)
    std = torch.tensor(config.PRETRAINED_STDS).view(3, 1, 1)
    
    # Move to same device as input
    mean = mean.to(img_tensor.device)
    std = std.to(img_tensor.device)
    
    # Denormalize
    img = img_tensor * std + mean
    
    # Clip to valid range
    return torch.clamp(img, 0, 1)


def tensor_to_numpy(img_tensor: torch.Tensor, denormalize: bool = True):
    """
    Convert image tensor to numpy array for visualization.
    
    Args:
        img_tensor: Image tensor (C, H, W)
        denormalize: Whether to denormalize first
        
    Returns:
        Numpy array (H, W, C)
    """
    import numpy as np
    
    if denormalize:
        img_tensor = denormalize_image(img_tensor)
    
    # Convert to numpy (H, W, C)
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    
    return np.clip(img_np, 0, 1)


class AugmentationVisualizer:
    """Helper class to visualize augmentations."""
    
    def __init__(self, transform):
        self.transform = transform
    
    def visualize(self, img_path: str, n_augmentations: int = 4):
        """
        Visualize multiple augmented versions of an image.
        
        Args:
            img_path: Path to image file
            n_augmentations: Number of augmented versions to show
        """
        import matplotlib.pyplot as plt
        from PIL import Image
        
        img = Image.open(img_path).convert('RGB')
        
        fig, axes = plt.subplots(1, n_augmentations + 1, figsize=(4 * (n_augmentations + 1), 4))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Augmented versions
        for i in range(n_augmentations):
            aug_img = self.transform(img)
            aug_np = tensor_to_numpy(aug_img)
            axes[i + 1].imshow(aug_np)
            axes[i + 1].set_title(f'Augmented {i + 1}')
            axes[i + 1].axis('off')
        
        plt.tight_layout()
        plt.show()
