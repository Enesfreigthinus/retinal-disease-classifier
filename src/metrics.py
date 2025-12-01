"""
Metrics module for Retinal Disease Classifier.
Contains evaluation metrics for multi-label classification.
"""

import torch
import numpy as np
from typing import Dict, Tuple


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate accuracy for multi-label classification.
    
    Args:
        y_pred: Predicted probabilities or logits
        y_true: Ground truth labels
        threshold: Classification threshold
        
    Returns:
        Accuracy value
    """
    y_pred = (y_pred > threshold).float()
    return (y_pred == y_true).float().mean(dim=1).mean().item()


def true_positive(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor, 
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Calculate true positives per class.
    
    Args:
        y_pred: Predicted probabilities or logits
        y_true: Ground truth labels
        threshold: Classification threshold
        
    Returns:
        True positives for each class
    """
    assert y_pred.shape == y_true.shape
    y_pred = (y_pred > threshold).float()
    return torch.sum((y_pred == 1) & (y_true == 1), dim=0)


def false_positive(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor, 
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Calculate false positives per class.
    
    Args:
        y_pred: Predicted probabilities or logits
        y_true: Ground truth labels
        threshold: Classification threshold
        
    Returns:
        False positives for each class
    """
    assert y_pred.shape == y_true.shape
    y_pred = (y_pred > threshold).float()
    return torch.sum((y_pred == 1) & (y_true == 0), dim=0)


def false_negative(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor, 
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Calculate false negatives per class.
    
    Args:
        y_pred: Predicted probabilities or logits
        y_true: Ground truth labels
        threshold: Classification threshold
        
    Returns:
        False negatives for each class
    """
    assert y_pred.shape == y_true.shape
    y_pred = (y_pred > threshold).float()
    return torch.sum((y_pred == 0) & (y_true == 1), dim=0)


def true_negative(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor, 
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Calculate true negatives per class.
    
    Args:
        y_pred: Predicted probabilities or logits
        y_true: Ground truth labels
        threshold: Classification threshold
        
    Returns:
        True negatives for each class
    """
    assert y_pred.shape == y_true.shape
    y_pred = (y_pred > threshold).float()
    return torch.sum((y_pred == 0) & (y_true == 0), dim=0)


def precision_score(tp: torch.Tensor, fp: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Calculate precision from TP and FP.
    
    Precision = TP / (TP + FP)
    
    Args:
        tp: True positives
        fp: False positives
        eps: Small value to avoid division by zero
        
    Returns:
        Precision for each class
    """
    return tp / (tp + fp + eps)


def recall_score(tp: torch.Tensor, fn: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Calculate recall from TP and FN.
    
    Recall = TP / (TP + FN)
    
    Args:
        tp: True positives
        fn: False negatives
        eps: Small value to avoid division by zero
        
    Returns:
        Recall for each class
    """
    return tp / (tp + fn + eps)


def f1_score(precision: torch.Tensor, recall: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Calculate F1 score from precision and recall.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        precision: Precision values
        recall: Recall values
        eps: Small value to avoid division by zero
        
    Returns:
        F1 score for each class
    """
    return 2 * (precision * recall) / (precision + recall + eps)


def specificity_score(tn: torch.Tensor, fp: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Calculate specificity from TN and FP.
    
    Specificity = TN / (TN + FP)
    
    Args:
        tn: True negatives
        fp: False positives
        eps: Small value to avoid division by zero
        
    Returns:
        Specificity for each class
    """
    return tn / (tn + fp + eps)


def compute_all_metrics(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor, 
    threshold: float = 0.5
) -> Dict[str, torch.Tensor]:
    """
    Compute all classification metrics.
    
    Args:
        y_pred: Predicted probabilities or logits
        y_true: Ground truth labels
        threshold: Classification threshold
        
    Returns:
        Dictionary with all metrics per class
    """
    tp = true_positive(y_pred, y_true, threshold)
    fp = false_positive(y_pred, y_true, threshold)
    fn = false_negative(y_pred, y_true, threshold)
    tn = true_negative(y_pred, y_true, threshold)
    
    precision = precision_score(tp, fp)
    recall = recall_score(tp, fn)
    f1 = f1_score(precision, recall)
    specificity = specificity_score(tn, fp)
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity
    }


def macro_average(metric: torch.Tensor) -> float:
    """
    Calculate macro average of a metric.
    
    Args:
        metric: Per-class metric values
        
    Returns:
        Macro average
    """
    return metric.mean().item()


def weighted_average(metric: torch.Tensor, weights: torch.Tensor) -> float:
    """
    Calculate weighted average of a metric.
    
    Args:
        metric: Per-class metric values
        weights: Weights for each class
        
    Returns:
        Weighted average
    """
    weights = weights / weights.sum()
    return (metric * weights).sum().item()


class MetricsTracker:
    """
    Tracks and accumulates metrics over batches.
    
    Usage:
        tracker = MetricsTracker(num_classes=43)
        for x, y in loader:
            y_pred = model(x)
            tracker.update(y_pred, y)
        results = tracker.compute()
    """
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all accumulators."""
        self.tp = torch.zeros(self.num_classes)
        self.fp = torch.zeros(self.num_classes)
        self.fn = torch.zeros(self.num_classes)
        self.tn = torch.zeros(self.num_classes)
        self.total_samples = 0
        self.total_acc = 0.0
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5):
        """
        Update metrics with a new batch.
        
        Args:
            y_pred: Predicted probabilities or logits
            y_true: Ground truth labels
            threshold: Classification threshold
        """
        y_pred = y_pred.detach().cpu()
        y_true = y_true.detach().cpu()
        
        self.tp += true_positive(y_pred, y_true, threshold)
        self.fp += false_positive(y_pred, y_true, threshold)
        self.fn += false_negative(y_pred, y_true, threshold)
        self.tn += true_negative(y_pred, y_true, threshold)
        
        batch_acc = accuracy(y_pred, y_true, threshold)
        self.total_acc += batch_acc * len(y_pred)
        self.total_samples += len(y_pred)
    
    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary with all metrics
        """
        precision = precision_score(self.tp, self.fp)
        recall = recall_score(self.tp, self.fn)
        f1 = f1_score(precision, recall)
        specificity = specificity_score(self.tn, self.fp)
        
        return {
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn,
            'tn': self.tn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'accuracy': self.total_acc / max(self.total_samples, 1),
            'macro_precision': macro_average(precision),
            'macro_recall': macro_average(recall),
            'macro_f1': macro_average(f1),
        }


def print_metrics_summary(
    metrics: Dict[str, torch.Tensor], 
    class_names: list = None,
    top_k: int = 10
):
    """
    Print a summary of metrics.
    
    Args:
        metrics: Dictionary of metrics
        class_names: Optional list of class names
        top_k: Number of top/bottom classes to show
    """
    print("=" * 70)
    print("METRICS SUMMARY")
    print("=" * 70)
    
    # Overall metrics
    print("\n📊 Overall Metrics:")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.3f}")
    print(f"  Macro Precision: {metrics.get('macro_precision', macro_average(metrics['precision'])):.3f}")
    print(f"  Macro Recall: {metrics.get('macro_recall', macro_average(metrics['recall'])):.3f}")
    print(f"  Macro F1: {metrics.get('macro_f1', macro_average(metrics['f1'])):.3f}")
    
    # Per-class metrics
    if class_names is not None:
        f1_scores = metrics['f1']
        
        # Top performing classes
        print(f"\n🏆 Top {top_k} Classes by F1 Score:")
        top_indices = torch.argsort(f1_scores, descending=True)[:top_k]
        for idx in top_indices:
            print(f"  {class_names[idx]}: F1={f1_scores[idx]:.3f}")
        
        # Bottom performing classes
        print(f"\n⚠️ Bottom {top_k} Classes by F1 Score:")
        bottom_indices = torch.argsort(f1_scores)[:top_k]
        for idx in bottom_indices:
            print(f"  {class_names[idx]}: F1={f1_scores[idx]:.3f}")
    
    # Classes with zero F1
    zero_f1 = (metrics['f1'] == 0).sum().item()
    if zero_f1 > 0:
        print(f"\n⚠️ Classes with F1=0: {zero_f1}")
    
    print("=" * 70)
