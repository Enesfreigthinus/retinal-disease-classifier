"""
Trainer module for Retinal Disease Classifier.
Contains training and evaluation loops.
"""

import os
import json
import torch
from torch import nn
from tqdm import tqdm
from typing import Dict, Callable, Optional

from . import config
from . import metrics as M


def train_epoch(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    gradient_clip: float = None
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: The model to train
        loader: Training data loader
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        scheduler: Optional learning rate scheduler
        gradient_clip: Max gradient norm for clipping
        
    Returns:
        Dictionary with training loss and accuracy
    """
    model.train()
    losses = []
    accuracies = []
    
    for x, y in tqdm(loader, desc=f'Epoch {epoch}'):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        
        optimizer.step()
        
        # Record metrics
        losses.append(loss.item())
        
        y_pred_cpu = y_pred.detach().cpu()
        y_cpu = y.detach().cpu()
        acc = M.accuracy(y_pred_cpu, y_cpu)
        accuracies.append(acc)
    
    # Update scheduler if provided
    if scheduler is not None:
        scheduler.step()
    
    return {
        'loss': torch.tensor(losses).mean().item(),
        'acc': torch.tensor(accuracies).mean().item()
    }


def evaluate(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: str
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: The model to evaluate
        loader: Data loader
        loss_fn: Loss function
        device: Device to evaluate on
        
    Returns:
        Dictionary with loss and accuracy
    """
    model.eval()
    losses = []
    accuracies = []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc='Evaluation'):
            x, y = x.to(device), y.to(device)
            
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            
            losses.append(loss.item())
            
            y_pred_cpu = y_pred.detach().cpu()
            y_cpu = y.detach().cpu()
            acc = M.accuracy(y_pred_cpu, y_cpu)
            accuracies.append(acc)
    
    return {
        'loss': torch.tensor(losses).mean().item(),
        'acc': torch.tensor(accuracies).mean().item()
    }


def evaluate_metrics(
    model: nn.Module,
    loader,
    device: str,
    num_classes: int = None
) -> Dict[str, torch.Tensor]:
    """
    Evaluate model and compute detailed metrics.
    
    Args:
        model: The model to evaluate
        loader: Data loader
        device: Device to evaluate on
        num_classes: Number of classes
        
    Returns:
        Dictionary with precision, recall, F1, etc.
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    
    tracker = M.MetricsTracker(num_classes)
    
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc='Computing metrics'):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            tracker.update(y_pred, y)
    
    return tracker.compute()


class Trainer:
    """
    Trainer class for managing the training loop.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: Optional LR scheduler
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = None,
        checkpoint_dir: str = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or config.DEVICE
        self.checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR
        
        # Training history
        self.train_history = []
        self.val_history = []
        self.best_val_loss = float('inf')
        self.best_model_path = ''
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train(
        self,
        num_epochs: int = None,
        early_stopping_patience: int = None,
        gradient_clip: float = None,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs (default from config)
            early_stopping_patience: Epochs without improvement before stopping
            gradient_clip: Max gradient norm for clipping
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history and best model path
        """
        if num_epochs is None:
            num_epochs = config.NUM_EPOCHS
        if early_stopping_patience is None:
            early_stopping_patience = config.EARLY_STOPPING_PATIENCE
        if gradient_clip is None:
            gradient_clip = config.GRADIENT_CLIP_NORM
        
        self.model = self.model.to(self.device)
        no_improvement_count = 0
        
        if verbose:
            print("=" * 60)
            print("TRAINING STARTED")
            print("=" * 60)
            print(f"Epochs: {num_epochs}")
            print(f"Early stopping patience: {early_stopping_patience}")
            print(f"Device: {self.device}")
            print("=" * 60)
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_res = train_epoch(
                self.model, self.train_loader, self.loss_fn,
                self.optimizer, self.device, epoch,
                self.scheduler, gradient_clip
            )
            
            # Validation
            val_res = evaluate(
                self.model, self.val_loader, self.loss_fn, self.device
            )
            
            # Record history
            self.train_history.append(train_res)
            self.val_history.append(val_res)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[-1]['lr']
            
            if verbose:
                print(f"\nEpoch {epoch}/{num_epochs} | LR: {current_lr:.2e}")
                print(f"  Train  - Loss: {train_res['loss']:.4f} | Acc: {train_res['acc']:.3f}")
                print(f"  Valid  - Loss: {val_res['loss']:.4f} | Acc: {val_res['acc']:.3f}")
            
            # Check for improvement
            val_loss = val_res['loss']
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_path = os.path.join(
                    self.checkpoint_dir, 
                    f'best_model_epoch_{epoch}.pth'
                )
                torch.save(self.model.state_dict(), self.best_model_path)
                no_improvement_count = 0
                if verbose:
                    print("  ★ New best model saved!")
            else:
                no_improvement_count += 1
                if verbose:
                    print(f"  (No improvement: {no_improvement_count}/{early_stopping_patience})")
            
            # Early stopping
            if no_improvement_count >= early_stopping_patience:
                if verbose:
                    print(f"\n⚠️ Early stopping after {epoch} epochs")
                break
        
        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING COMPLETED")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"Best model saved to: {self.best_model_path}")
            print("=" * 60)
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_loss': self.best_val_loss,
            'best_model_path': self.best_model_path
        }
    
    def load_best_model(self):
        """Load the best model from training."""
        if self.best_model_path and os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path))
            print(f"Loaded best model from: {self.best_model_path}")
        else:
            print("No best model checkpoint found.")
    
    def save_history(self, path: str = None):
        """Save training history to JSON file."""
        if path is None:
            path = os.path.join(self.checkpoint_dir, 'training_history.json')
        
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'best_val_loss': self.best_val_loss,
            'best_model_path': self.best_model_path
        }
        
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved to: {path}")


def collect_predictions(
    model: nn.Module,
    loader,
    device: str
) -> tuple:
    """
    Collect all predictions and labels from a data loader.
    
    Args:
        model: The model
        loader: Data loader
        device: Device to use
        
    Returns:
        Tuple of (predictions, labels) as tensors
    """
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc='Collecting predictions'):
            x = x.to(device)
            y_pred = model(x)
            y_pred = torch.sigmoid(y_pred)  # Convert logits to probabilities
            
            all_preds.append(y_pred.cpu())
            all_labels.append(y.cpu())
    
    return torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0)
