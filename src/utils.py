"""
Utils module for Retinal Disease Classifier.
Contains utility functions and helper classes.
"""

import os
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from typing import Tuple, List

from . import config


class LRFinder:
    """
    Learning Rate Finder using the technique from the paper:
    "Cyclical Learning Rates for Training Neural Networks" (Smith, 2017)
    
    Gradually increases learning rate and tracks loss to find optimal LR.
    
    Args:
        model: Model to test
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        
    Usage:
        lr_finder = LRFinder(model, optimizer, criterion, device)
        lrs, losses = lr_finder.range_test(train_loader)
        lr_finder.plot(lrs, losses)
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        criterion: nn.Module, 
        device: str
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Save initial model state
        self._initial_state = os.path.join(config.OUTPUT_DIR, 'lr_finder_init.pt')
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        torch.save(model.state_dict(), self._initial_state)
    
    def _train_batch(self, iterator) -> float:
        """Train on a single batch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        x, y = iterator.get_batch()
        x = x.to(self.device)
        y = y.to(self.device)
        
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def range_test(
        self,
        train_loader,
        start_lr: float = None,
        end_lr: float = None,
        num_iter: int = None,
        smooth_f: float = 0.05,
        diverge_th: float = 5.0
    ) -> Tuple[List[float], List[float]]:
        """
        Perform learning rate range test.
        
        Args:
            train_loader: Training data loader
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations
            smooth_f: Smoothing factor for loss
            diverge_th: Threshold for detecting divergence
            
        Returns:
            Tuple of (learning_rates, losses)
        """
        if start_lr is None:
            start_lr = config.LR_FINDER_START_LR
        if end_lr is None:
            end_lr = config.LR_FINDER_END_LR
        if num_iter is None:
            num_iter = config.LR_FINDER_NUM_ITER
        
        # Set starting LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = start_lr
        
        lrs = []
        losses = []
        best_loss = float('inf')
        
        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)
        iterator = IteratorWrapper(train_loader)
        
        for iteration in tqdm(range(num_iter), desc='LR Range Test'):
            loss = self._train_batch(iterator)
            lrs.append(lr_scheduler.get_last_lr()[0])
            
            # Update LR
            lr_scheduler.step()
            
            # Smooth loss
            if iteration > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]
            
            if loss < best_loss:
                best_loss = loss
            
            losses.append(loss)
            
            # Check for divergence
            if loss > diverge_th * best_loss:
                print(f"Stopping early at iteration {iteration}, loss diverged")
                break
        
        # Restore initial model state
        self.model.load_state_dict(torch.load(self._initial_state))
        
        return lrs, losses
    
    def suggest_lr(self, lrs: List[float], losses: List[float]) -> float:
        """
        Suggest optimal learning rate.
        
        The suggested LR is the point with lowest loss divided by 10.
        
        Args:
            lrs: Learning rates from range test
            losses: Corresponding losses
            
        Returns:
            Suggested learning rate
        """
        min_idx = losses.index(min(losses))
        return lrs[min_idx] / 10


class ExponentialLR(_LRScheduler):
    """
    Exponential learning rate scheduler for LR finder.
    
    Increases LR exponentially from base_lr to end_lr over num_iter iterations.
    """
    
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        end_lr: float, 
        num_iter: int, 
        last_epoch: int = -1
    ):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        cur_iter = self.last_epoch
        r = cur_iter / self.num_iter
        return [base_lr * ((self.end_lr / base_lr) ** r) for base_lr in self.base_lrs]


class IteratorWrapper:
    """
    Wrapper for data loader that allows infinite iteration.
    
    Automatically resets the iterator when exhausted.
    """
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._iterator = iter(data_loader)
    
    def __next__(self):
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.data_loader)
            batch = next(self._iterator)
        return batch
    
    def get_batch(self):
        return next(self)


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")


def print_gpu_info():
    """Print GPU information if available."""
    if torch.cuda.is_available():
        print("=" * 60)
        print("GPU INFORMATION")
        print("=" * 60)
        print(f"CUDA Available: True")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nDevice {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
        
        print("=" * 60)
    else:
        print("CUDA not available. Using CPU.")


def format_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def count_samples_per_class(labels: torch.Tensor) -> torch.Tensor:
    """
    Count number of positive samples per class.
    
    Args:
        labels: Label tensor of shape (N, C)
        
    Returns:
        Tensor with counts per class
    """
    return labels.sum(dim=0)


def get_class_weights(labels: torch.Tensor, method: str = 'inverse') -> torch.Tensor:
    """
    Calculate class weights for imbalanced data.
    
    Args:
        labels: Label tensor of shape (N, C)
        method: Weighting method ('inverse', 'sqrt_inverse', 'effective')
        
    Returns:
        Tensor with weight per class
    """
    counts = count_samples_per_class(labels).float()
    n_samples = len(labels)
    
    if method == 'inverse':
        weights = n_samples / (counts + 1)
    elif method == 'sqrt_inverse':
        weights = torch.sqrt(n_samples / (counts + 1))
    elif method == 'effective':
        # Effective number of samples method
        beta = 0.999
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize
    weights = weights / weights.sum() * len(weights)
    
    return weights


class EarlyStopping:
    """
    Early stopping handler.
    
    Args:
        patience: Number of epochs without improvement before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
            return True
        
        return False
    
    def reset(self):
        """Reset the early stopping counter."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False


def load_image(path: str):
    """
    Load an image from file.
    
    Args:
        path: Path to image file
        
    Returns:
        PIL Image
    """
    from PIL import Image
    return Image.open(path).convert('RGB')


def save_predictions(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    output_path: str
):
    """
    Save predictions to CSV file.
    
    Args:
        predictions: Prediction tensor
        labels: Ground truth tensor
        class_names: List of class names
        output_path: Path to save CSV
    """
    import pandas as pd
    
    pred_np = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
    label_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    
    # Create DataFrame
    data = {}
    for i, name in enumerate(class_names):
        data[f'{name}_true'] = label_np[:, i]
        data[f'{name}_pred'] = pred_np[:, i]
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
