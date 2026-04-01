"""
Fairness-Aware Training for Phase 3: Loss Curve Exploration

This module provides fairness-aware training functions that integrate
fairness penalties into Ditto's training loop using weighted loss combination.

Key Features:
- Fairness-aware train_step with combined loss
- Enhanced evaluation with fairness metrics
- Compatible with existing Ditto model architecture
- Supports alpha-weighted loss: (1-alpha) × accuracy + alpha × fairness
"""

import os
import sys
import torch
import torch.nn as nn
import sklearn.metrics as metrics
from typing import Dict, Any

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ditto'))
sys.path.insert(0, os.path.dirname(__file__))  # Add fairness_em to path

from torch.utils import data
from transformers import AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from ditto_light.ditto import DittoModel

# Import fairness utilities
from fairness_utils import (
    calculate_fairness_loss,
    calculate_fairness_loss_with_ema,
    evaluate_fairness_metrics,
    FairnessEMATracker
)


def fairness_aware_train_step(train_iter, model, optimizer, scheduler, hp,
                               alpha_fairness=0.0, ema_tracker=None, class_weights=None):
    """
    Perform fairness-aware training step with combined loss.

    This function extends Ditto's train_step to incorporate fairness penalties.
    The total loss is: (1 - alpha) × accuracy_loss + alpha × fairness_loss

    Args:
        train_iter: DataLoader iterator with enhanced batches
        model: DittoModel instance
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
        hp: Hyperparameters namespace
        alpha_fairness: Weight for fairness penalty (0.0 = pure accuracy, 1.0 = pure fairness)
        ema_tracker: Optional FairnessEMATracker for stable fairness calculation (Phase B)

    Returns:
        Dictionary with average losses for the epoch
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model.train()

    total_loss_sum = 0.0
    accuracy_loss_sum = 0.0
    fairness_loss_sum = 0.0
    num_batches = 0

    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()

        # Unpack enhanced batch (includes sensitive attributes)
        if len(batch) == 3:
            # No data augmentation: (x, y, sensitive_attrs)
            x, y, sensitive_attrs = batch
            prediction = model(x)
        elif len(batch) == 4:
            # With data augmentation: (x1, x2, y, sensitive_attrs)
            x1, x2, y, sensitive_attrs = batch
            prediction = model(x1, x2)
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        # Calculate accuracy loss (standard cross-entropy)
        accuracy_loss = criterion(prediction, y.to(model.device))

        # Calculate fairness loss if alpha > 0
        if alpha_fairness > 0.0:
            if ema_tracker is not None:
                # Phase B: Use EMA-based fairness loss for stability
                fairness_loss = calculate_fairness_loss_with_ema(
                    prediction,
                    y.to(model.device),
                    sensitive_attrs,
                    ema_tracker
                )
            else:
                # Original batch-level fairness loss
                fairness_loss = calculate_fairness_loss(
                    prediction,
                    y.to(model.device),
                    sensitive_attrs,
                    fairness_metric="ppvp"
                )
        else:
            fairness_loss = torch.tensor(0.0, device=model.device)

        # Combined loss with alpha weighting
        total_loss = (1 - alpha_fairness) * accuracy_loss + alpha_fairness * fairness_loss

        # Backward pass
        if hp.fp16:
            try:
                from apex import amp
                with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            except ImportError:
                total_loss.backward()
        else:
            total_loss.backward()

        optimizer.step()
        scheduler.step()

        # Accumulate losses for logging
        total_loss_sum += total_loss.item()
        accuracy_loss_sum += accuracy_loss.item()
        fairness_loss_sum += fairness_loss.item() if isinstance(fairness_loss, torch.Tensor) else fairness_loss

        num_batches += 1

        # Periodic logging
        if i % 10 == 0:
            print(f"step: {i}, total_loss: {total_loss.item():.4f}, "
                  f"acc_loss: {accuracy_loss.item():.4f}, "
                  f"fair_loss: {fairness_loss.item() if isinstance(fairness_loss, torch.Tensor) else fairness_loss:.4f}")

        del total_loss, accuracy_loss, fairness_loss

    # Return average losses
    return {
        'total_loss': total_loss_sum / num_batches,
        'accuracy_loss': accuracy_loss_sum / num_batches,
        'fairness_loss': fairness_loss_sum / num_batches
    }


def fairness_evaluate(model, iterator, threshold=None):
    """
    Enhanced evaluation with both accuracy and fairness metrics.

    Args:
        model: DittoModel instance
        iterator: DataLoader iterator
        threshold: Optional classification threshold

    Returns:
        If threshold is None: (f1_score, best_threshold, fairness_metrics)
        If threshold is provided: (f1_score, fairness_metrics)
    """
    model.eval()

    all_probs = []
    all_y = []
    all_sensitive_attrs = []

    with torch.no_grad():
        for batch in iterator:
            # Unpack batch (with or without data augmentation)
            if len(batch) == 3:
                x, y, sensitive_attrs = batch
            elif len(batch) == 4:
                x, _, y, sensitive_attrs = batch  # Use first input if DA enabled
            else:
                raise ValueError(f"Unexpected batch format")

            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy().tolist())
            all_y.extend(y.cpu().numpy().tolist())
            all_sensitive_attrs.extend(sensitive_attrs)

    # Calculate accuracy metrics
    if threshold is not None:
        # Use provided threshold
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(all_y, pred)

        # Calculate fairness metrics
        fairness_metrics = evaluate_fairness_metrics(
            all_probs, all_y, all_sensitive_attrs, threshold
        )

        return f1, fairness_metrics
    else:
        # Find best threshold
        best_th = 0.5
        best_f1 = 0.0

        for th in torch.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            if new_f1 > best_f1:
                best_f1 = new_f1
                best_th = th.item()

        # Calculate fairness metrics at best threshold
        fairness_metrics = evaluate_fairness_metrics(
            all_probs, all_y, all_sensitive_attrs, best_th
        )

        return best_f1, best_th, fairness_metrics


def fairness_aware_train(trainset, validset, testset, run_tag, hp,
                         alpha_fairness=0.0, use_ema=True, ema_beta=0.9,
                         checkpoint_dir=None, weight_method='sqrt'):
    """
    Main fairness-aware training function.

    This function orchestrates the complete training pipeline with fairness awareness.
    It's a drop-in replacement for Ditto's train() function with fairness support.

    Args:
        trainset: FairnessDittoDataset for training
        validset: FairnessDittoDataset for validation
        testset: FairnessDittoDataset for testing
        run_tag: Unique identifier for this training run
        hp: Hyperparameters namespace
        alpha_fairness: Weight for fairness penalty (default: 0.0 for baseline)
        use_ema: Whether to use EMA-based fairness calculation (Phase B, default: True)
        ema_beta: EMA decay factor (default: 0.9)

    Returns:
        Dictionary with training results and paths
    """
    print(f"\n{'='*70}")
    print(f"Fairness-Aware Training: alpha = {alpha_fairness}")
    print(f"Run Tag: {run_tag}")
    if use_ema and alpha_fairness > 0:
        print(f"Using EMA-based fairness (Phase B): beta = {ema_beta}")
    print(f"{'='*70}\n")

    # Use FairnessDittoDataset's pad function for batching
    padder = trainset.pad

    # Create DataLoaders with enhanced batching
    train_iter = data.DataLoader(
        dataset=trainset,
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=padder
    )
    valid_iter = data.DataLoader(
        dataset=validset,
        batch_size=hp.batch_size * 16,
        shuffle=False,
        num_workers=0,
        collate_fn=padder
    )
    test_iter = data.DataLoader(
        dataset=testset,
        batch_size=hp.batch_size * 16,
        shuffle=False,
        num_workers=0,
        collate_fn=padder
    )

    # Initialize model, optimizer, and scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = DittoModel(device=device, lm=hp.lm, alpha_aug=hp.alpha_aug)
    if device == 'cuda':
        model = model.cuda()

    optimizer = AdamW(model.parameters(), lr=hp.lr)

    # FP16 training (if APEX available and requested)
    if hp.fp16:
        try:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
            print("FP16 training enabled with APEX")
        except ImportError:
            print("Warning: FP16 requested but APEX not available. Using FP32.")

    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_steps
    )

    # TensorBoard logging
    writer = SummaryWriter(log_dir=hp.logdir)

    # Create EMA tracker for Phase B fairness calculation
    ema_tracker = None
    if use_ema and alpha_fairness > 0:
        ema_tracker = FairnessEMATracker(beta=ema_beta)
        print(f"EMA tracker initialized for fairness calculation")

    # Compute class weights to handle class imbalance
    import math
    num_pos = sum(1 for l in trainset.labels if l == 1)
    num_neg = len(trainset.labels) - num_pos
    if num_pos > 0 and num_neg > 0:
        raw_pos = len(trainset.labels) / (2.0 * num_pos)
        raw_neg = len(trainset.labels) / (2.0 * num_neg)
        
        if weight_method == 'sqrt':
            weight_pos = math.sqrt(raw_pos)
            weight_neg = math.sqrt(raw_neg)
            method_name = "sqrt-dampened"
        else:
            weight_pos = raw_pos
            weight_neg = raw_neg
            method_name = "normal"
            
        class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float32, device=device)
        print(f"Class weights ({method_name}): neg={weight_neg:.4f}, pos={weight_pos:.4f} "
              f"(raw_ratio={raw_pos/raw_neg:.1f}:1, current_ratio={weight_pos/weight_neg:.1f}:1, "
              f"pos_rate={num_pos/len(trainset.labels):.4f})")
    else:
        class_weights = None

    # Training loop
    best_dev_f1 = 0.0
    best_test_f1 = 0.0
    best_fairness_metrics = None
    best_epoch = 0

    for epoch in range(1, hp.n_epochs + 1):
        print(f"\n--- Epoch {epoch}/{hp.n_epochs} ---")

        # Train
        epoch_losses = fairness_aware_train_step(
            train_iter, model, optimizer, scheduler, hp, alpha_fairness,
            ema_tracker=ema_tracker, class_weights=class_weights
        )

        # Log EMA statistics if available
        if ema_tracker is not None:
            ema_stats = ema_tracker.get_stats()
            print(f"  EMA Stats: groups={ema_stats['groups_tracked']}, "
                  f"dp_disparity={ema_stats['dp_disparity']:.4f}, "
                  f"ppvp_disparity={ema_stats['ppvp_disparity']:.4f}")

        # Evaluate
        model.eval()
        dev_f1, th, dev_fairness = fairness_evaluate(model, valid_iter)
        test_f1, test_fairness = fairness_evaluate(model, test_iter, threshold=th)

        # Display results
        print(f"\nEpoch {epoch} Results:")
        print(f"  Validation F1: {dev_f1:.4f}")
        print(f"  Test F1: {test_f1:.4f}")
        print(f"  PPVP Disparity: {test_fairness['ppvp_disparity']:.4f}")
        print(f"  Fair Groups: {test_fairness['num_fair_groups']}/{test_fairness['total_groups']}")

        # Track best model
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            best_fairness_metrics = test_fairness
            best_epoch = epoch

            # Save model checkpoint
            if hp.save_model:
                ckpt_dir = checkpoint_dir if checkpoint_dir else os.path.join(hp.logdir, 'Compas')
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f'model_alpha_{alpha_fairness:.2f}.pt')
                ckpt = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'alpha_fairness': alpha_fairness,
                    'best_f1': best_test_f1,
                    'fairness_metrics': best_fairness_metrics
                }
                torch.save(ckpt, ckpt_path)
                print(f"  Model saved: {ckpt_path}")

        # TensorBoard logging
        writer.add_scalars(run_tag + '/losses', epoch_losses, epoch)
        writer.add_scalars(run_tag + '/metrics', {
            'dev_f1': dev_f1,
            'test_f1': test_f1,
            'ppvp_disparity': test_fairness['ppvp_disparity']
        }, epoch)

    writer.close()

    # Final summary
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Test F1: {best_test_f1:.4f}")
    print(f"  Final PPVP Disparity: {best_fairness_metrics['ppvp_disparity']:.4f}")
    print(f"  Fair Groups: {best_fairness_metrics['num_fair_groups']}/{best_fairness_metrics['total_groups']}")
    print(f"{'='*70}\n")

    # Clean up model to free memory
    result = {
        'best_f1': best_test_f1,
        'best_epoch': best_epoch,
        'fairness_metrics': best_fairness_metrics,
        'alpha': alpha_fairness,
        'model_path': os.path.join(
            checkpoint_dir if checkpoint_dir else os.path.join(hp.logdir, 'Compas'),
            f'model_alpha_{alpha_fairness:.2f}.pt'
        )
    }

    # Explicit cleanup to prevent memory leaks between experiments
    del model
    del optimizer
    del scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


if __name__ == "__main__":
    print("fairness_train.py - Fairness-Aware Training Module")
    print("This module is designed to be imported by phase3_experiment.py")
    print("For standalone testing, use phase3_experiment.py")
