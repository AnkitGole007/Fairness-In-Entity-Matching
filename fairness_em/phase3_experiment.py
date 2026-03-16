"""
Phase 3: Loss Curve Exploration Experiment

This is the main experimental script that orchestrates the complete Phase 3 pipeline:
1. Runs training with multiple alpha values (0.0 to 1.0)
2. Collects comprehensive results for each alpha
3. Generates the accuracy-fairness trade-off data
4. Saves results for visualization

Run artifact contract:
    Every run is saved under results/phase3/<run_id>/ where run_id is a
    datetime stamp with millisecond precision (YYYY-MM-DD_HH-mm-ss-SSS).
    Inside each run folder:
        checkpoints/          - model .pt files per alpha
        visuals/              - plots (populated by visualize_results.py)
        training_log.md       - structured markdown log
        experiment_config.json
        phase3_loss_curve_results.csv
        phase3_results_intermediate.csv

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
import subprocess
import time
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ditto'))
sys.path.insert(0, os.path.dirname(__file__))

from fairness_dataset import FairnessDittoDataset
from fairness_train import fairness_aware_train
from visualize_results import run_analysis


# ---------------------------------------------------------------------------
# Logging Utilities
# ---------------------------------------------------------------------------

class TeeLogger:
    """
    Custom logger to redirect stdout/stderr to both the terminal and a file.
    """
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# ---------------------------------------------------------------------------
# Run ID generation
# ---------------------------------------------------------------------------

def generate_run_id():
    """
    Generate a run_id in datetime format with millisecond precision.
    Format: YYYY-MM-DD_HH-mm-ss-SSS
    If the folder already exists, sleep 1 ms and retry (up to 10 times).
    """
    for _ in range(10):
        now = datetime.now()
        run_id = now.strftime("%Y-%m-%d_%H-%M-%S-") + f"{now.microsecond // 1000:03d}"
        return run_id
    raise RuntimeError("Could not generate a unique run_id after 10 attempts")


def get_git_commit_hash():
    """Return the short git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


# ---------------------------------------------------------------------------
# Run folder creation
# ---------------------------------------------------------------------------

def create_run_folders(base_dir, run_id):
    """
    Create the run artifact folder tree. Raises if the folder already exists.

    Returns:
        dict with resolved absolute paths for each subfolder and key file.
    """
    run_dir = os.path.abspath(os.path.join(base_dir, run_id))

    if os.path.exists(run_dir):
        raise FileExistsError(f"Run folder already exists: {run_dir}")

    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    visuals_dir = os.path.join(run_dir, 'visuals')

    os.makedirs(checkpoints_dir, exist_ok=False)
    os.makedirs(visuals_dir, exist_ok=False)

    return {
        'run_dir': run_dir,
        'checkpoints_dir': checkpoints_dir,
        'visuals_dir': visuals_dir,
        'training_log': os.path.join(run_dir, 'training_log.md'),
        'config_file': os.path.join(run_dir, 'experiment_config.json'),
        'results_csv': os.path.join(run_dir, 'phase3_loss_curve_results.csv'),
        'intermediate_csv': os.path.join(run_dir, 'phase3_results_intermediate.csv'),
    }


# ---------------------------------------------------------------------------
# Training log (Markdown)
# ---------------------------------------------------------------------------

def write_training_log(paths, config, run_id, results, command_used, start_time, end_time):
    """
    Write a structured training_log.md into the run folder.

    Sections:
        1. Run metadata
        2. Config
        3. Outputs summary
        4. Visuals index
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
    commit = get_git_commit_hash()

    duration = end_time - start_time
    duration_str = f"{duration:.1f}s ({duration / 60:.1f} min)"

    # Collect checkpoint file names
    ckpt_files = []
    if os.path.isdir(paths['checkpoints_dir']):
        ckpt_files = sorted(os.listdir(paths['checkpoints_dir']))

    # Collect visual file names
    visual_files = []
    if os.path.isdir(paths['visuals_dir']):
        visual_files = sorted(os.listdir(paths['visuals_dir']))

    lines = []
    lines.append(f"# Phase 3 Training Log")
    lines.append("")

    # --- Section 1: Run Metadata ---
    lines.append("## 1. Run Metadata")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| **Run ID** | `{run_id}` |")
    lines.append(f"| **Start Time** | {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')} |")
    lines.append(f"| **End Time** | {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')} |")
    lines.append(f"| **Duration** | {duration_str} |")
    lines.append(f"| **Device** | {device} ({gpu_name}) |")
    lines.append(f"| **Commit Hash** | `{commit}` |")
    lines.append(f"| **Command** | `{command_used}` |")
    lines.append("")

    # --- Section 2: Config ---
    lines.append("## 2. Configuration")
    lines.append("")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| **Alpha Values** | {config.alpha_values} |")
    lines.append(f"| **Epochs** | {config.n_epochs} |")
    lines.append(f"| **Batch Size** | {config.batch_size} |")
    lines.append(f"| **Learning Rate** | {config.lr} |")
    lines.append(f"| **Seed (Run ID)** | {config.run_id} |")
    lines.append(f"| **Language Model** | {config.lm} |")
    lines.append(f"| **Max Length** | {config.max_len} |")
    lines.append(f"| **Dataset Size** | {'Full' if config.dataset_size is None else config.dataset_size} |")
    lines.append(f"| **Dataset** | {config.task} |")
    lines.append(f"| **Weight Method** | sqrt-dampened inverse frequency |")
    lines.append(f"| **FP16** | {config.fp16} |")
    lines.append("")

    # --- Section 3: Outputs Summary ---
    lines.append("## 3. Outputs Summary")
    lines.append("")

    if results is not None and len(results) > 0:
        lines.append("### Final Metrics per Alpha")
        lines.append("")
        lines.append("| Alpha | F1 Score | PPVP Disparity | Fairness Rate | Best Epoch |")
        lines.append("|-------|----------|----------------|---------------|------------|")
        for r in results:
            lines.append(
                f"| {r['alpha']:.1f} | {r['f1_score']:.4f} | "
                f"{r['ppvp_disparity']:.4f} | {r['fairness_rate']:.2f} | "
                f"{r['best_epoch']} |"
            )
        lines.append("")

        # Best result
        best = max(results, key=lambda r: r['f1_score'])
        lines.append(f"**Best F1:** {best['f1_score']:.4f} at alpha={best['alpha']:.1f} (epoch {best['best_epoch']})")
        lines.append("")
    else:
        lines.append("*No results — all experiments failed.*")
        lines.append("")

    lines.append("### Checkpoint Files")
    lines.append("")
    if ckpt_files:
        for f in ckpt_files:
            lines.append(f"- `checkpoints/{f}`")
    else:
        lines.append("- *No checkpoints saved.*")
    lines.append("")

    # --- Section 4: Visuals Index ---
    lines.append("## 4. Visuals Index")
    lines.append("")
    if visual_files:
        for f in visual_files:
            lines.append(f"- `visuals/{f}`")
    else:
        lines.append("- *No visuals generated yet. Run `visualize_results.py` to populate.*")
    lines.append("")

    # Write atomically: write to temp, then rename
    tmp_path = paths['training_log'] + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    os.replace(tmp_path, paths['training_log'])
    print(f"Training log written: {paths['training_log']}")


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

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
        if args is not None and hasattr(args, 'alpha_values') and args.alpha_values:
            self.alpha_values = args.alpha_values
        else:
            self.alpha_values = [0.0, 0.5, 1.0]

        # Dataset size limit (None = full dataset, or specify number for testing)
        self.dataset_size = None if args is None else args.dataset_size

        # Create top-level output directory (run subfolder created later)
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


def run_single_alpha_experiment(alpha, config, run_id, checkpoint_dir):
    """
    Run one complete training + evaluation cycle for a given alpha value.

    Args:
        alpha: Fairness penalty weight (0.0 to 1.0)
        config: ExperimentConfig instance
        run_id: Random seed for reproducibility
        checkpoint_dir: Directory to save model checkpoints

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
    start_time = datetime.now()
    result = fairness_aware_train(
        train_dataset,
        valid_dataset,
        test_dataset,
        run_tag,
        hp,
        alpha_fairness=alpha,
        use_ema=True,
        ema_beta=0.9,
        checkpoint_dir=checkpoint_dir
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
    All outputs are saved under results/phase3/<run_id>/.

    Args:
        config: ExperimentConfig instance

    Returns:
        DataFrame with comprehensive results
    """
    # ---- Generate run_id and create folder tree ----
    run_id = generate_run_id()
    base_dir = config.results_dir

    # Retry with new run_id if folder exists (collision guard)
    for attempt in range(10):
        try:
            paths = create_run_folders(base_dir, run_id)
            break
        except FileExistsError:
            time.sleep(0.002)
            run_id = generate_run_id()
    else:
        raise RuntimeError("Could not create a unique run folder after 10 attempts")

    command_used = ' '.join(sys.argv)
    start_time = time.time()

    # Start capturing terminal output
    terminal_log_path = os.path.join(paths['run_dir'], 'terminal_output.log')
    logger = TeeLogger(terminal_log_path)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = logger
    sys.stderr = logger

    try:
        print("\n" + "="*80)
        print(" PHASE 3: LOSS CURVE EXPLORATION")
        print("="*80)
        print(f"\n  Run ID: {run_id}")
        print(f"  Run Folder: {paths['run_dir']}")
        print(f"\nExperiment Configuration:")
        print(f"  Task: {config.task}")
        print(f"  Language Model: {config.lm}")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  Epochs: {config.n_epochs}")
        print(f"  Alpha Values: {config.alpha_values}")
        print(f"  Dataset Size: {config.dataset_size if config.dataset_size else 'Full'}")
        print(f"\nResults will be saved to: {paths['run_dir']}")
        print("="*80 + "\n")

        # Save configuration
        config_dict = config.to_dict()
        config_dict['run_id'] = run_id
        config_dict['commit_hash'] = get_git_commit_hash()
        config_dict['command'] = command_used

        tmp_cfg = paths['config_file'] + '.tmp'
        with open(tmp_cfg, 'w') as f:
            json.dump(config_dict, f, indent=2)
        os.replace(tmp_cfg, paths['config_file'])
        print(f"Configuration saved: {paths['config_file']}\n")

        # Run experiments for each alpha value
        results = []

        for idx, alpha in enumerate(config.alpha_values, 1):
            print(f"\n{'*'*80}")
            print(f"* PROGRESS: {idx}/{len(config.alpha_values)} - alpha = {alpha}")
            print(f"{'*'*80}")

            try:
                result = run_single_alpha_experiment(
                    alpha, config, config.run_id, paths['checkpoints_dir']
                )
                results.append(result)

                # Save intermediate results after each experiment (atomic write)
                intermediate_df = pd.DataFrame(results)
                tmp_csv = paths['intermediate_csv'] + '.tmp'
                intermediate_df.to_csv(tmp_csv, index=False)
                os.replace(tmp_csv, paths['intermediate_csv'])
                print(f"Intermediate results saved: {paths['intermediate_csv']}")

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

        end_time = time.time()

        # Convert to DataFrame and save final results
        if not results:
            print("\n[ERROR] No experiments completed successfully!")
            write_training_log(paths, config, run_id, results, command_used, start_time, end_time)
            return None

        results_df = pd.DataFrame(results)

        # Sort by alpha for clarity
        results_df = results_df.sort_values('alpha').reset_index(drop=True)

        # Save comprehensive results (atomic write)
        tmp_csv = paths['results_csv'] + '.tmp'
        results_df.to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, paths['results_csv'])

        print("\n" + "="*80)
        print(" PHASE 3 EXPERIMENT COMPLETE!")
        print("="*80)
        print(f"\n  Run ID: {run_id}")
        print(f"  Run Folder: {paths['run_dir']}")
        print(f"\nFinal Results Summary:")
        print(results_df[['alpha', 'f1_score', 'ppvp_disparity', 'fairness_rate']].to_string(index=False))
        print(f"\nResults saved to: {paths['results_csv']}")
        print("="*80 + "\n")

        # Write training log
        write_training_log(paths, config, run_id, results, command_used, start_time, end_time)

        # Automatically generate visualizations
        print("\nGenerating visualizations...")
        run_analysis(run_dir=paths['run_dir'])

    finally:
        # Reset terminal output and close logger
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger.close()

    return results_df


def main():
    """Main entry point for Phase 3 experiments"""
    parser = argparse.ArgumentParser(description='Phase 3: Loss Curve Exploration')

    parser.add_argument('--alpha_values', type=float, nargs='+',
                       help='Alpha values to test (default: 0.0 0.5 1.0)')
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
        print("\nNext step: Run visualize_results.py to generate plots")
    else:
        print("\nExperiment failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
