"""
Model module for Retinal Disease Classifier.
Contains model definitions and loading utilities.
"""

import torch
from torch import nn
from torchvision import models

from . import config


def create_convnext_model(
    num_classes: int = None,
    pretrained: bool = True,
    freeze_features: bool = False
) -> nn.Module:
    """
    Create a ConvNeXt-Tiny model for multi-label classification.
    
    Args:
        num_classes: Number of output classes (default from config)
        pretrained: Whether to use pretrained weights
        freeze_features: Whether to freeze feature extractor
        
    Returns:
        ConvNeXt model with modified classifier
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    
    # Load pretrained model
    weights = config.PRETRAINED_WEIGHTS if pretrained else None
    model = models.convnext_tiny(weights=weights)
    
    # Modify classifier for multi-label classification
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    # Optionally freeze feature extractor
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
    
    return model


def get_model_params(model: nn.Module) -> dict:
    """
    Get parameter groups for discriminative fine-tuning.
    
    Uses smaller learning rate for feature extractor compared to classifier.
    
    Args:
        model: The model
        
    Returns:
        Dictionary with parameter groups
    """
    return [
        {
            'params': model.features.parameters(), 
            'lr': config.LR_FEATURE_EXTRACTOR
        },
        {
            'params': model.classifier.parameters(), 
            'lr': config.LR_CLASSIFIER
        }
    ]


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count model parameters.
    
    Args:
        model: The model
        trainable_only: Count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str = None) -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        model: Model architecture
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        
    Returns:
        Model with loaded weights
    """
    if device is None:
        device = config.DEVICE
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    return model


def save_checkpoint(model: nn.Module, path: str):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        path: Path to save checkpoint
    """
    torch.save(model.state_dict(), path)


def get_optimizer(
    model: nn.Module,
    lr: float = None,
    weight_decay: float = None,
    use_discriminative_lr: bool = True
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with optional discriminative learning rates.
    
    Args:
        model: The model
        lr: Learning rate (default from config)
        weight_decay: Weight decay (default from config)
        use_discriminative_lr: Use different LR for features vs classifier
        
    Returns:
        Optimizer
    """
    if lr is None:
        lr = config.LEARNING_RATE
    if weight_decay is None:
        weight_decay = config.WEIGHT_DECAY
    
    if use_discriminative_lr:
        params = get_model_params(model)
    else:
        params = model.parameters()
    
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int = None,
    scheduler_type: str = 'cosine'
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: The optimizer
        num_epochs: Number of training epochs
        scheduler_type: Type of scheduler ('cosine', 'step', 'plateau')
        
    Returns:
        LR scheduler
    """
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS
    
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs, 
            eta_min=1e-6
        )
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=num_epochs // 3, 
            gamma=0.1
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=3
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def print_model_summary(model: nn.Module):
    """Print model architecture summary."""
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")
    print(f"Output classes: {config.NUM_CLASSES}")
    print("=" * 60)
    print("\nClassifier architecture:")
    print(model.classifier)
    print("=" * 60)


class ModelFactory:
    """Factory class for creating different model architectures."""
    
    SUPPORTED_MODELS = {
        'convnext_tiny': models.convnext_tiny,
        'convnext_small': models.convnext_small,
        'convnext_base': models.convnext_base,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b3': models.efficientnet_b3,
    }
    
    @classmethod
    def create(
        cls, 
        model_name: str, 
        num_classes: int = None, 
        pretrained: bool = True
    ) -> nn.Module:
        """
        Create a model by name.
        
        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            
        Returns:
            Model instance
        """
        if num_classes is None:
            num_classes = config.NUM_CLASSES
            
        if model_name not in cls.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Supported: {list(cls.SUPPORTED_MODELS.keys())}"
            )
        
        # Create base model
        weights = 'IMAGENET1K_V1' if pretrained else None
        model = cls.SUPPORTED_MODELS[model_name](weights=weights)
        
        # Modify classifier based on model type
        if 'convnext' in model_name:
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif 'resnet' in model_name:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif 'efficientnet' in model_name:
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        
        return model
    
    @classmethod
    def list_models(cls) -> list:
        """Return list of supported model names."""
        return list(cls.SUPPORTED_MODELS.keys())
