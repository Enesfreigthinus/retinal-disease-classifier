"""
Visualization module for Retinal Disease Classifier.
Contains plotting functions for analysis and presentation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Dict, List, Optional
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix

from . import config


def plot_training_curves(
    train_history: List[Dict],
    val_history: List[Dict],
    save_path: Optional[str] = None
):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        train_history: List of training metrics per epoch
        val_history: List of validation metrics per epoch
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_history) + 1)
    
    # Loss curve
    train_losses = [r['loss'] for r in train_history]
    val_losses = [r['loss'] for r in val_history]
    
    axes[0].plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    axes[0].plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    train_accs = [r['acc'] for r in train_history]
    val_accs = [r['acc'] for r in val_history]
    
    axes[1].plot(epochs, train_accs, 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
    axes[1].plot(epochs, val_accs, 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_class_distribution(
    class_counts: dict,
    save_path: Optional[str] = None
):
    """
    Plot horizontal bar chart of class distribution.
    
    Args:
        class_counts: Dictionary or Series of class counts
        save_path: Optional path to save the figure
    """
    if hasattr(class_counts, 'sort_values'):
        class_counts = class_counts.sort_values(ascending=True)
    
    plt.figure(figsize=(10, 10))
    
    if hasattr(class_counts, 'plot'):
        class_counts.plot(kind='barh', color='orange', edgecolor='black')
    else:
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        plt.barh(classes, counts, color='orange', edgecolor='black')
    
    plt.title('Distribution of Disease Classes', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Count', fontsize=12)
    plt.ylabel('Disease Class', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_metrics_bar(
    metrics: Dict[str, torch.Tensor],
    class_names: List[str],
    top_k: int = 15,
    save_path: Optional[str] = None
):
    """
    Plot precision, recall, and F1 as grouped bar chart.
    
    Args:
        metrics: Dictionary with precision, recall, f1 tensors
        class_names: List of class names
        top_k: Number of top classes to show
        save_path: Optional path to save the figure
    """
    # Sort by F1 score
    f1_scores = metrics['f1'].numpy() if isinstance(metrics['f1'], torch.Tensor) else metrics['f1']
    top_indices = np.argsort(f1_scores)[-top_k:][::-1]
    
    precision = metrics['precision'].numpy()[top_indices] if isinstance(metrics['precision'], torch.Tensor) else np.array(metrics['precision'])[top_indices]
    recall = metrics['recall'].numpy()[top_indices] if isinstance(metrics['recall'], torch.Tensor) else np.array(metrics['recall'])[top_indices]
    f1 = f1_scores[top_indices]
    top_names = [class_names[i] for i in top_indices]
    
    x = np.arange(len(top_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(16, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db', edgecolor='black')
    bars3 = ax.bar(x + width, f1, width, label='F1 Score', color='#e74c3c', edgecolor='black')
    
    ax.set_xlabel('Disease Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Precision, Recall & F1 Score - Top {top_k} Classes', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_names, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_confusion_matrices(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_names: List[str],
    indices: List[int] = None,
    title: str = "Confusion Matrices",
    save_path: Optional[str] = None,
    cmap: str = 'Blues'
):
    """
    Plot confusion matrices for selected classes.
    
    Args:
        y_true: Ground truth labels
        y_pred: Binary predictions
        class_names: List of class names
        indices: Indices of classes to plot (default: first 10)
        title: Figure title
        save_path: Optional path to save the figure
        cmap: Colormap to use
    """
    if indices is None:
        indices = list(range(min(10, len(class_names))))
    
    y_true_np = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_np = y_pred.numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    
    mcm = multilabel_confusion_matrix(y_true_np, y_pred_np)
    
    n_cols = 5
    n_rows = (len(indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if len(indices) == 1 else axes
    
    for idx, (class_idx, ax) in enumerate(zip(indices, axes)):
        cm = mcm[class_idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    xticklabels=['Pred: 0', 'Pred: 1'],
                    yticklabels=['True: 0', 'True: 1'],
                    annot_kws={'size': 12})
        ax.set_title(f'{class_names[class_idx]}', fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # Hide empty subplots
    for idx in range(len(indices), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_roc_curves(
    y_true: torch.Tensor,
    y_pred_proba: torch.Tensor,
    class_names: List[str],
    indices: List[int] = None,
    save_path: Optional[str] = None
):
    """
    Plot ROC curves for selected classes.
    
    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities
        class_names: List of class names
        indices: Indices of classes to plot (default: first 5)
        save_path: Optional path to save the figure
    """
    if indices is None:
        indices = list(range(min(5, len(class_names))))
    
    y_true_np = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_np = y_pred_proba.numpy() if isinstance(y_pred_proba, torch.Tensor) else y_pred_proba
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(indices)))
    
    for idx, (class_idx, color) in enumerate(zip(indices, colors)):
        fpr, tpr, _ = roc_curve(y_true_np[:, class_idx], y_pred_np[:, class_idx])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, 
                label=f'{class_names[class_idx]} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_metrics_heatmap(
    metrics: Dict[str, torch.Tensor],
    class_names: List[str],
    save_path: Optional[str] = None
):
    """
    Plot heatmap of all metrics across classes.
    
    Args:
        metrics: Dictionary with precision, recall, f1 tensors
        class_names: List of class names
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(20, 6))
    
    precision = metrics['precision'].numpy() if isinstance(metrics['precision'], torch.Tensor) else metrics['precision']
    recall = metrics['recall'].numpy() if isinstance(metrics['recall'], torch.Tensor) else metrics['recall']
    f1 = metrics['f1'].numpy() if isinstance(metrics['f1'], torch.Tensor) else metrics['f1']
    
    metrics_data = np.stack([precision, recall, f1])
    
    sns.heatmap(metrics_data, annot=False, cmap='RdYlGn', ax=ax,
                xticklabels=class_names, yticklabels=['Precision', 'Recall', 'F1'],
                vmin=0, vmax=1, cbar_kws={'label': 'Score'})
    
    ax.set_title('Performance Metrics Heatmap - All Classes', fontsize=14, fontweight='bold')
    ax.set_xlabel('Disease Class', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_class_vs_performance(
    class_counts,
    f1_scores: torch.Tensor,
    class_names: List[str],
    save_path: Optional[str] = None
):
    """
    Plot class frequency vs model performance.
    
    Args:
        class_counts: Class sample counts
        f1_scores: F1 scores per class
        class_names: List of class names
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if hasattr(class_counts, 'values'):
        counts = class_counts.values
    else:
        counts = np.array(list(class_counts.values()))
    
    f1_np = f1_scores.numpy() if isinstance(f1_scores, torch.Tensor) else f1_scores
    
    x = np.arange(len(class_names))
    width = 0.35
    
    ax2 = ax.twinx()
    
    bars1 = ax.bar(x - width/2, counts, width, label='Sample Count', color='#3498db', alpha=0.7)
    bars2 = ax2.bar(x + width/2, f1_np, width, label='F1 Score', color='#e74c3c', alpha=0.7)
    
    ax.set_xlabel('Disease Class', fontsize=12)
    ax.set_ylabel('Sample Count', fontsize=12, color='#3498db')
    ax2.set_ylabel('F1 Score', fontsize=12, color='#e74c3c')
    ax.set_title('Class Distribution vs Model Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_sample_predictions(
    model,
    dataset,
    class_names: List[str],
    device: str,
    n_samples: int = 6,
    save_path: Optional[str] = None
):
    """
    Plot sample predictions with ground truth.
    
    Args:
        model: Trained model
        dataset: Dataset to sample from
        class_names: List of class names
        device: Device to use
        n_samples: Number of samples to show
        save_path: Optional path to save the figure
    """
    from .transforms import tensor_to_numpy
    import random
    
    sample_indices = random.sample(range(len(dataset)), n_samples)
    
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()
    
    model.eval()
    
    for idx, sample_idx in enumerate(sample_indices):
        img, true_label = dataset[sample_idx]
        
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device))
            pred = torch.sigmoid(pred).cpu().squeeze()
        
        # Denormalize image
        img_display = tensor_to_numpy(img)
        
        # Get disease names
        true_diseases = [class_names[i] for i in range(len(class_names)) if true_label[i] == 1]
        pred_diseases = [class_names[i] for i in range(len(class_names)) if pred[i] > 0.5]
        
        axes[idx].imshow(img_display)
        axes[idx].axis('off')
        
        true_str = ', '.join(true_diseases[:3]) if true_diseases else 'Healthy'
        pred_str = ', '.join(pred_diseases[:3]) if pred_diseases else 'Healthy'
        
        if len(true_diseases) > 3:
            true_str += f' +{len(true_diseases) - 3}'
        if len(pred_diseases) > 3:
            pred_str += f' +{len(pred_diseases) - 3}'
        
        is_correct = set(true_diseases) == set(pred_diseases)
        color = 'green' if is_correct else 'red'
        
        axes[idx].set_title(f'True: {true_str}\nPred: {pred_str}',
                           fontsize=9, color=color, fontweight='bold')
    
    # Hide empty subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_lr_finder(
    lrs: List[float],
    losses: List[float],
    skip_start: int = 5,
    skip_end: int = 5,
    save_path: Optional[str] = None
):
    """
    Plot learning rate finder results.
    
    Args:
        lrs: Learning rates tested
        losses: Corresponding losses
        skip_start: Number of initial points to skip
        skip_end: Number of final points to skip
        save_path: Optional path to save the figure
    """
    if skip_end == 0:
        lrs = lrs[skip_start:]
        losses = losses[skip_start:]
    else:
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(lrs, losses, linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Learning Rate Finder', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    
    # Mark suggested LR
    min_idx = np.argmin(losses)
    suggested_lr = lrs[min_idx] / 10
    ax.axvline(x=suggested_lr, color='r', linestyle='--', 
               label=f'Suggested LR: {suggested_lr:.2e}')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def save_all_plots(
    train_history: List[Dict],
    val_history: List[Dict],
    metrics: Dict[str, torch.Tensor],
    class_names: List[str],
    class_counts,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    output_dir: str = None
):
    """
    Save all analysis plots.
    
    Args:
        train_history: Training history
        val_history: Validation history
        metrics: Evaluation metrics
        class_names: List of class names
        class_counts: Class sample counts
        y_true: Ground truth labels
        y_pred: Binary predictions
        output_dir: Directory to save plots
    """
    if output_dir is None:
        output_dir = config.PLOTS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating and saving all plots...")
    
    # Training curves
    plot_training_curves(
        train_history, val_history,
        save_path=os.path.join(output_dir, 'training_curves.png')
    )
    
    # Class distribution
    plot_class_distribution(
        class_counts,
        save_path=os.path.join(output_dir, 'class_distribution.png')
    )
    
    # Metrics bar chart
    plot_metrics_bar(
        metrics, class_names,
        save_path=os.path.join(output_dir, 'metrics_bar.png')
    )
    
    # Metrics heatmap
    plot_metrics_heatmap(
        metrics, class_names,
        save_path=os.path.join(output_dir, 'metrics_heatmap.png')
    )
    
    # Class vs performance
    plot_class_vs_performance(
        class_counts, metrics['f1'], class_names,
        save_path=os.path.join(output_dir, 'class_vs_performance.png')
    )
    
    print(f"\n✅ All plots saved to: {output_dir}")
