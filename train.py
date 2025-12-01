"""
Main training script for Retinal Disease Classifier.

Usage:
    python train.py [--config CONFIG_PATH] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
    
Example:
    python train.py --epochs 30 --batch-size 64
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src import dataset
from src import transforms
from src import model
from src import losses
from src import trainer
from src import metrics
from src import visualization
from src import utils


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Retinal Disease Classifier')
    
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=config.EARLY_STOPPING_PATIENCE,
                        help='Early stopping patience')
    parser.add_argument('--loss', type=str, default='focal',
                        choices=['focal', 'bce', 'weighted_bce', 'asymmetric'],
                        help='Loss function type')
    parser.add_argument('--use-sampler', action='store_true', default=True,
                        help='Use weighted random sampler')
    parser.add_argument('--advanced-aug', action='store_true', default=True,
                        help='Use advanced data augmentation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default=config.OUTPUT_DIR,
                        help='Output directory')
    parser.add_argument('--find-lr', action='store_true',
                        help='Run learning rate finder before training')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    utils.set_seed(args.seed)
    
    # Print GPU info
    utils.print_gpu_info()
    
    # Create output directories
    config.create_output_dirs()
    
    # Print configuration
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Loss function: {args.loss}")
    print(f"Use weighted sampler: {args.use_sampler}")
    print(f"Advanced augmentation: {args.advanced_aug}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60 + "\n")
    
    # Get transforms
    if args.advanced_aug:
        train_transform = transforms.get_advanced_train_transform()
        print("✓ Using advanced data augmentation")
    else:
        train_transform = transforms.get_basic_train_transform()
        print("✓ Using basic data augmentation")
    
    test_transform = transforms.get_test_transform()
    
    # Setup datasets
    print("\n📂 Loading datasets...")
    data_setup = dataset.setup_datasets(train_transform, test_transform)
    
    print(f"  Training samples: {data_setup['num_train']}")
    print(f"  Validation samples: {data_setup['num_val']}")
    print(f"  Test samples: {data_setup['num_test']}")
    print(f"  Number of classes: {len(data_setup['class_names'])}")
    
    # Create data loaders
    if args.use_sampler:
        sampler = dataset.create_weighted_sampler(data_setup['sample_weights'])
        print("✓ Using weighted random sampler for class imbalance")
    else:
        sampler = None
    
    train_loader, val_loader, test_loader = dataset.create_data_loaders(
        data_setup['train_data'],
        data_setup['val_data'],
        data_setup['test_data'],
        batch_size=args.batch_size,
        sampler=sampler
    )
    
    # Create model
    print("\n🧠 Creating model...")
    net = model.create_convnext_model()
    net = net.to(config.DEVICE)
    model.print_model_summary(net)
    
    # Create optimizer with discriminative learning rates
    optimizer = model.get_optimizer(net, lr=args.lr)
    print(f"✓ Optimizer: AdamW with discriminative LR")
    
    # Create scheduler
    scheduler = model.get_scheduler(optimizer, num_epochs=args.epochs)
    print(f"✓ Scheduler: Cosine Annealing")
    
    # Create loss function
    pos_weight = data_setup['pos_weight'].to(config.DEVICE)
    loss_fn = losses.get_loss_function(
        loss_type=args.loss,
        pos_weight=pos_weight,
        device=config.DEVICE
    )
    losses.print_loss_summary(args.loss, pos_weight)
    
    # Optional: Run LR Finder
    if args.find_lr:
        print("\n🔍 Running Learning Rate Finder...")
        lr_finder = utils.LRFinder(net, optimizer, loss_fn, config.DEVICE)
        lrs, lr_losses = lr_finder.range_test(train_loader)
        suggested_lr = lr_finder.suggest_lr(lrs, lr_losses)
        print(f"Suggested LR: {suggested_lr:.2e}")
        
        visualization.plot_lr_finder(
            lrs, lr_losses,
            save_path=os.path.join(args.output_dir, 'lr_finder.png')
        )
        
        # Update optimizer with suggested LR
        for param_group in optimizer.param_groups:
            if 'features' in str(param_group.get('name', '')):
                param_group['lr'] = suggested_lr / 10
            else:
                param_group['lr'] = suggested_lr
    
    # Create trainer
    train_manager = trainer.Trainer(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.DEVICE,
        checkpoint_dir=config.CHECKPOINT_DIR
    )
    
    # Train
    print("\n🚀 Starting training...")
    training_result = train_manager.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        gradient_clip=config.GRADIENT_CLIP_NORM
    )
    
    # Save training history
    train_manager.save_history()
    
    # Load best model for evaluation
    train_manager.load_best_model()
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_metrics = trainer.evaluate_metrics(
        net, test_loader, config.DEVICE, config.NUM_CLASSES
    )
    
    metrics.print_metrics_summary(test_metrics, data_setup['class_names'])
    
    # Collect predictions for visualization
    print("\n📈 Generating visualizations...")
    y_pred_proba, y_true = trainer.collect_predictions(net, test_loader, config.DEVICE)
    y_pred_binary = (y_pred_proba > 0.5).float()
    
    # Generate and save all plots
    visualization.plot_training_curves(
        training_result['train_history'],
        training_result['val_history'],
        save_path=os.path.join(config.PLOTS_DIR, 'training_curves.png')
    )
    
    visualization.plot_metrics_bar(
        test_metrics,
        data_setup['class_names'],
        save_path=os.path.join(config.PLOTS_DIR, 'metrics_bar.png')
    )
    
    visualization.plot_metrics_heatmap(
        test_metrics,
        data_setup['class_names'],
        save_path=os.path.join(config.PLOTS_DIR, 'metrics_heatmap.png')
    )
    
    visualization.plot_roc_curves(
        y_true, y_pred_proba,
        data_setup['class_names'],
        save_path=os.path.join(config.PLOTS_DIR, 'roc_curves.png')
    )
    
    # Confusion matrices for top 10 classes
    class_counts = dataset.calculate_class_frequencies(data_setup['train_label_df'])
    top_10_indices = class_counts.nlargest(10).index.tolist()
    top_10_indices = [data_setup['class_names'].index(c) for c in top_10_indices]
    
    visualization.plot_confusion_matrices(
        y_true, y_pred_binary,
        data_setup['class_names'],
        indices=top_10_indices,
        title='Confusion Matrices - Top 10 Classes',
        save_path=os.path.join(config.PLOTS_DIR, 'confusion_matrices_top10.png')
    )
    
    # Sample predictions
    visualization.plot_sample_predictions(
        net, data_setup['test_data'],
        data_setup['class_names'],
        config.DEVICE,
        save_path=os.path.join(config.PLOTS_DIR, 'sample_predictions.png')
    )
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best model saved to: {training_result['best_model_path']}")
    print(f"Best validation loss: {training_result['best_val_loss']:.4f}")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.3f}")
    print(f"Plots saved to: {config.PLOTS_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
