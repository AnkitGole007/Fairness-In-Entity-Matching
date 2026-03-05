"""
Phase 3: Loss Curve Exploration Experiment

This is the main experimental script that orchestrates the complete Phase 3 pipeline:
1. Runs training with multiple alpha values (0.0 to 1.0)
2. Collects comprehensive results for each alpha
3. Generates the accuracy-fairness trade-off data
4. Saves results for visualization

Usage:
    python phase3_experiment.py [--alpha_values 0.0 0.3 0.5 0.7 1.0] [--epochs 20] [--batch_size 64]
"""

import os
import sys
import json
import argparse
import pandas as pd
import torch
import numpy as np
import random
import gc
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ditto'))
sys.path.insert(0, os.path.dirname(__file__))

from fairness_dataset import FairnessDittoDataset
from fairness_train import fairness_aware_train


class ExperimentConfig:
    """Configuration management for Phase 3 experiments"""

    def __init__(self, args=None):
        """
        Initialize experiment configuration.

        Args:
            args: Argument namespace from argparse (optional)
        """
        # Dataset configuration
        self.task = "Compas"
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'ditto', 'data', 'Compas')
        self.trainset_path = os.path.join(self.data_dir, 'train.txt')
        self.validset_path = os.path.join(self.data_dir, 'valid.txt')
        self.testset_path = os.path.join(self.data_dir, 'test.txt')

        # Model configuration
        self.lm = 'distilbert'
        self.max_len = 256
        self.alpha_aug = 0.8

        # Training configuration (can be overridden by args)
        self.batch_size = 64 if args is None else args.batch_size
        self.lr = 3e-5
        self.n_epochs = 20 if args is None else args.n_epochs
        self.run_id = 0

        # Experiment configuration
        self.save_model = True
        self.fp16 = False  # Disable FP16 for stability
        self.logdir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'phase3')
        self.results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'phase3')

        # Alpha values for loss curve exploration
        # Per 02/19 meeting: Start with three-option experiment
        # alpha=0.0 (pure accuracy), alpha=0.5 (balanced), alpha=1.0 (pure fairness)
        if args is not None and hasattr(args, 'alpha_values') and args.alpha_values:
            self.alpha_values = args.alpha_values
        else:
            # Three-option experiment per Professor's guidance
            self.alpha_values = [0.0, 0.5, 1.0]

        # Dataset size limit (None = full dataset, or specify number for testing)
        self.dataset_size = None if args is None else args.dataset_size

        # Create output directories
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def to_dict(self):
        """Convert config to dictionary for saving"""
        return {
            'task': self.task,
            'lm': self.lm,
            'max_len': self.max_len,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'n_epochs': self.n_epochs,
            'alpha_values': self.alpha_values,
            'dataset_size': self.dataset_size
        }


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_alpha_experiment(alpha, config, run_id):
    """
    Run one complete training + evaluation cycle for a given alpha value.

    Args:
        alpha: Fairness penalty weight (0.0 to 1.0)
        config: ExperimentConfig instance
        run_id: Random seed for reproducibility

    Returns:
        Dictionary with comprehensive results
    """
    print(f"\n{'#'*80}")
    print(f"# EXPERIMENT: alpha = {alpha}")
    print(f"# Run ID: {run_id}")
    print(f"{'#'*80}\n")

    # Set seed for reproducibility
    set_seed(run_id)

    # Load datasets with FairnessDittoDataset
    print("Loading datasets...")
    train_dataset = FairnessDittoDataset(
        config.trainset_path,
        lm=config.lm,
        max_len=config.max_len,
        size=config.dataset_size
    )
    valid_dataset = FairnessDittoDataset(
        config.validset_path,
        lm=config.lm,
        max_len=config.max_len
    )
    test_dataset = FairnessDittoDataset(
        config.testset_path,
        lm=config.lm,
        max_len=config.max_len
    )

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Valid: {len(valid_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # Display dataset statistics
    train_stats = train_dataset.get_sensitive_attribute_stats()
    print(f"\nTraining Data Demographics:")
    print(f"  Same demographic: {train_stats['same_group']}")
    print(f"  Mixed demographics: {train_stats['mixed_group']}")
    print(f"  Distribution: {train_stats['group_distribution']}")

    # Create run tag
    run_tag = f"Compas_alpha_{alpha:.2f}_id{run_id}"

    # Create hyperparameters namespace (matching Ditto's format)
    class HP:
        pass

    hp = HP()
    hp.batch_size = config.batch_size
    hp.max_len = config.max_len
    hp.lr = config.lr
    hp.n_epochs = config.n_epochs
    hp.lm = config.lm
    hp.alpha_aug = config.alpha_aug
    hp.fp16 = config.fp16
    hp.save_model = config.save_model
    hp.logdir = config.logdir
    hp.task = config.task

    # Train model with fairness awareness
    # Phase B: Use EMA-based fairness calculation for stability
    start_time = datetime.now()
    result = fairness_aware_train(
        train_dataset,
        valid_dataset,
        test_dataset,
        run_tag,
        hp,
        alpha_fairness=alpha,
        use_ema=True,  # Phase B: EMA-based fairness for stable disparity calculation
        ema_beta=0.9   # EMA decay factor (90% history, 10% current batch)
    )
    training_time = (datetime.now() - start_time).total_seconds()

    # Compile comprehensive results
    result_dict = {
        'alpha': alpha,
        'run_id': run_id,
        'f1_score': result['best_f1'],
        'best_epoch': result['best_epoch'],
        'training_time_seconds': training_time,
        'ppvp_disparity': result['fairness_metrics']['ppvp_disparity'],
        'num_fair_groups': result['fairness_metrics']['num_fair_groups'],
        'total_groups': result['fairness_metrics']['total_groups'],
        'fairness_rate': result['fairness_metrics'].get('fairness_rate',
            result['fairness_metrics']['num_fair_groups'] / max(result['fairness_metrics']['total_groups'], 1)),
        'model_path': result['model_path'],
        'run_tag': run_tag
    }

    # Add per-group PPVP values
    for group, ppvp in result['fairness_metrics']['ppvp_by_group'].items():
        result_dict[f'ppvp_{group}'] = ppvp

    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE: alpha = {alpha}")
    print(f"  F1 Score: {result_dict['f1_score']:.4f}")
    print(f"  PPVP Disparity: {result_dict['ppvp_disparity']:.4f}")
    print(f"  Fairness Rate: {result_dict['fairness_rate']:.2%}")
    print(f"  Training Time: {training_time:.1f}s")
    print(f"{'='*80}\n")

    return result_dict


def run_loss_curve_exploration(config):
    """
    Main experimental orchestrator for Phase 3.

    Runs the complete alpha sweep and generates the loss curve data.

    Args:
        config: ExperimentConfig instance

    Returns:
        DataFrame with comprehensive results
    """
    print("\n" + "="*80)
    print(" PHASE 3: LOSS CURVE EXPLORATION")
    print("="*80)
    print(f"\nExperiment Configuration:")
    print(f"  Task: {config.task}")
    print(f"  Language Model: {config.lm}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Alpha Values: {config.alpha_values}")
    print(f"  Dataset Size: {config.dataset_size if config.dataset_size else 'Full'}")
    print(f"\nResults will be saved to: {config.results_dir}")
    print("="*80 + "\n")

    # Save configuration
    config_path = os.path.join(config.results_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Configuration saved: {config_path}\n")

    # Run experiments for each alpha value
    results = []

    for idx, alpha in enumerate(config.alpha_values, 1):
        print(f"\n{'*'*80}")
        print(f"* PROGRESS: {idx}/{len(config.alpha_values)} - alpha = {alpha}")
        print(f"{'*'*80}")

        try:
            result = run_single_alpha_experiment(alpha, config, config.run_id)
            results.append(result)

            # Save intermediate results after each experiment
            intermediate_df = pd.DataFrame(results)
            intermediate_path = os.path.join(config.results_dir, 'phase3_results_intermediate.csv')
            intermediate_df.to_csv(intermediate_path, index=False)
            print(f"Intermediate results saved: {intermediate_path}")

        except Exception as e:
            print(f"\n[ERROR] Experiment failed for alpha = {alpha}: {e}")
            import traceback
            traceback.print_exc()
            continue

        finally:
            # Clean up GPU memory between experiments
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # Convert to DataFrame and save final results
    if not results:
        print("\n[ERROR] No experiments completed successfully!")
        return None

    results_df = pd.DataFrame(results)

    # Sort by alpha for clarity
    results_df = results_df.sort_values('alpha').reset_index(drop=True)

    # Save comprehensive results
    results_path = os.path.join(config.results_dir, 'phase3_loss_curve_results.csv')
    results_df.to_csv(results_path, index=False)

    print("\n" + "="*80)
    print(" PHASE 3 EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"\nFinal Results Summary:")
    print(results_df[['alpha', 'f1_score', 'ppvp_disparity', 'fairness_rate']].to_string(index=False))
    print(f"\nResults saved to: {results_path}")
    print("="*80 + "\n")

    return results_df


def main():
    """Main entry point for Phase 3 experiments"""
    parser = argparse.ArgumentParser(description='Phase 3: Loss Curve Exploration')

    parser.add_argument('--alpha_values', type=float, nargs='+',
                       help='Alpha values to test (default: 0.0 0.1 0.2 0.3 0.5 0.7 0.8 0.9 1.0)')
    parser.add_argument('--epochs', dest='n_epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--dataset_size', type=int, default=None,
                       help='Limit dataset size for quick testing (default: None = full dataset)')

    args = parser.parse_args()

    # Create configuration
    config = ExperimentConfig(args)

    # Run experiments
    results_df = run_loss_curve_exploration(config)

    if results_df is not None:
        print("\nExperiment completed successfully!")
        print(f"Results saved to: {config.results_dir}")
        print("\nNext step: Run visualize_results.py to generate plots")
    else:
        print("\nExperiment failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
