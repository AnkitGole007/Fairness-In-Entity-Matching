"""
Fairness Utility Functions for Phase 3: Loss Curve Exploration

This module provides core fairness calculation functions that can be reused
across different components of the fairness-aware training system.

Key Functions:
- extract_sensitive_attribute: Parse Ditto COL/VAL format
- calculate_ppvp_disparity: Compute PPVP fairness metric
- calculate_fairness_loss: Main loss function for training
- group_by_demographic: Helper for batch processing
- FairnessEMATracker: Running average tracker for stable fairness calculation (Phase B)
"""

import re
import torch
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any


class FairnessEMATracker:
    """
    Exponential Moving Average (EMA) tracker for stable fairness calculation.

    Phase B Implementation: Addresses the pointwise vs group-wise challenge by
    maintaining running averages of group statistics across batches.

    Instead of calculating PPVP from a single batch (which may lack representation),
    this tracker accumulates soft TP and FP counts over time, providing stable
    estimates for fairness disparity calculation.

    Attributes:
        beta: EMA decay factor (0.9 = 90% old, 10% new)
        group_soft_tp: Running average of soft true positives per group
        group_soft_fp: Running average of soft false positives per group
        group_counts: Number of samples seen per group (for diagnostics)
        initialized: Whether each group has been initialized
    """

    def __init__(self, beta: float = 0.9):
        """
        Initialize the EMA tracker.

        Args:
            beta: EMA decay factor. Higher = more history, lower = more responsive.
                  Default 0.9 means each batch contributes 10% to the running average.
        """
        self.beta = beta
        self.group_soft_tp = {}  # group_name -> EMA of soft TP
        self.group_soft_fp = {}  # group_name -> EMA of soft FP
        self.group_mean_prob = {}  # group_name -> EMA of mean prediction prob (for DP)
        self.group_counts = {}   # group_name -> total samples seen
        self.initialized = {}    # group_name -> bool

    def update(self, probs: torch.Tensor, labels: torch.Tensor,
               sensitive_attrs: List[Tuple[str, str]]) -> None:
        """
        Update EMA statistics with a new batch.

        Args:
            probs: Prediction probabilities for class 1 [batch_size]
            labels: True labels [batch_size]
            sensitive_attrs: List of (left_attr, right_attr) tuples
        """
        # Group indices by demographics
        group_indices = defaultdict(list)

        for i, (left_attr, right_attr) in enumerate(sensitive_attrs):
            if left_attr == right_attr and left_attr is not None:
                group_key = left_attr
                group_indices[group_key].append(i)

        # Update EMA for each group present in this batch
        for group_name, indices in group_indices.items():
            if len(indices) == 0:
                continue

            # Extract group data
            group_probs = probs[indices].detach()  # Detach for EMA update
            group_labels = labels[indices].float()

            # Calculate batch soft TP and FP
            batch_soft_tp = (group_probs * group_labels).sum().item()
            batch_soft_fp = (group_probs * (1 - group_labels)).sum().item()
            batch_count = len(indices)
            batch_mean_prob = group_probs.mean().item()  # For Demographic Parity

            # Update EMA
            if group_name not in self.initialized or not self.initialized[group_name]:
                # First time seeing this group - initialize
                self.group_soft_tp[group_name] = batch_soft_tp
                self.group_soft_fp[group_name] = batch_soft_fp
                self.group_mean_prob[group_name] = batch_mean_prob
                self.group_counts[group_name] = batch_count
                self.initialized[group_name] = True
            else:
                # EMA update: new = beta * old + (1 - beta) * current
                self.group_soft_tp[group_name] = (
                    self.beta * self.group_soft_tp[group_name] +
                    (1 - self.beta) * batch_soft_tp
                )
                self.group_soft_fp[group_name] = (
                    self.beta * self.group_soft_fp[group_name] +
                    (1 - self.beta) * batch_soft_fp
                )
                self.group_mean_prob[group_name] = (
                    self.beta * self.group_mean_prob[group_name] +
                    (1 - self.beta) * batch_mean_prob
                )
                self.group_counts[group_name] += batch_count

    def get_ppvp_disparity(self) -> float:
        """
        Calculate PPVP disparity from accumulated EMA statistics.

        Returns:
            Disparity value (max PPVP - min PPVP), or 0.0 if < 2 groups
        """
        ppvp_values = []

        for group_name in self.initialized:
            if not self.initialized[group_name]:
                continue

            soft_tp = self.group_soft_tp[group_name]
            soft_fp = self.group_soft_fp[group_name]

            # PPVP = soft_TP / (soft_TP + soft_FP)
            if soft_tp + soft_fp > 1e-8:
                ppvp = soft_tp / (soft_tp + soft_fp)
                ppvp_values.append(ppvp)

        if len(ppvp_values) < 2:
            return 0.0

        return max(ppvp_values) - min(ppvp_values)

    def get_dp_disparity(self) -> float:
        """
        Calculate Demographic Parity (DP) disparity from EMA statistics.

        DP measures the difference in mean prediction probability across groups.
        Unlike PPVP, DP is non-degenerate even when predictions are all-negative.

        Returns:
            Disparity value (max mean_prob - min mean_prob), or 0.0 if < 2 groups
        """
        mean_probs = []
        for group_name in self.initialized:
            if self.initialized[group_name] and group_name in self.group_mean_prob:
                mean_probs.append(self.group_mean_prob[group_name])

        if len(mean_probs) < 2:
            return 0.0

        return max(mean_probs) - min(mean_probs)

    def get_group_ppvp(self) -> Dict[str, float]:
        """Get PPVP for each tracked group."""
        result = {}
        for group_name in self.initialized:
            if not self.initialized[group_name]:
                continue
            soft_tp = self.group_soft_tp[group_name]
            soft_fp = self.group_soft_fp[group_name]
            if soft_tp + soft_fp > 1e-8:
                result[group_name] = soft_tp / (soft_tp + soft_fp)
            else:
                result[group_name] = 0.0
        return result

    def reset(self):
        """Reset all accumulated statistics."""
        self.group_soft_tp = {}
        self.group_soft_fp = {}
        self.group_mean_prob = {}
        self.group_counts = {}
        self.initialized = {}

    def get_stats(self) -> Dict[str, Any]:
        """Get diagnostic statistics."""
        return {
            'groups_tracked': list(self.initialized.keys()),
            'total_samples': sum(self.group_counts.values()),
            'group_counts': dict(self.group_counts),
            'group_ppvp': self.get_group_ppvp(),
            'ppvp_disparity': self.get_ppvp_disparity(),
            'dp_disparity': self.get_dp_disparity(),
            'group_mean_prob': dict(self.group_mean_prob)
        }


def extract_sensitive_attribute(text_sequence: str, attribute_name: str = "Ethnic_Code_Text") -> Optional[str]:
    """
    Extract demographic information from Ditto's COL/VAL format.

    Args:
        text_sequence: String in Ditto format, e.g., "COL id VAL 123 COL Ethnic_Code_Text VAL African-American"
        attribute_name: Name of the sensitive attribute to extract (default: "Ethnic_Code_Text")

    Returns:
        The extracted attribute value (e.g., "African-American"), or None if not found

    Example:
        >>> text = "COL Ethnic_Code_Text VAL African-American COL name VAL John"
        >>> extract_sensitive_attribute(text)
        'African-American'
    """
    # Pattern: COL <attribute_name> VAL <value>
    # Match everything after VAL until next COL or tab/end of string
    pattern = rf'COL\s+{re.escape(attribute_name)}\s+VAL\s+([^\t]+?)(?:\s+COL|\t|$)'
    match = re.search(pattern, text_sequence)

    if match:
        return match.group(1).strip()
    return None


def extract_sensitive_attributes_from_pair(left_text: str, right_text: str,
                                          attribute_name: str = "Ethnic_Code_Text") -> Tuple[Optional[str], Optional[str]]:
    """
    Extract sensitive attributes from both sides of an entity pair.

    Args:
        left_text: Left entity in Ditto COL/VAL format
        right_text: Right entity in Ditto COL/VAL format
        attribute_name: Name of the sensitive attribute

    Returns:
        Tuple of (left_attribute, right_attribute)
    """
    left_attr = extract_sensitive_attribute(left_text, attribute_name)
    right_attr = extract_sensitive_attribute(right_text, attribute_name)
    return (left_attr, right_attr)


def group_by_demographic(predictions: torch.Tensor, labels: torch.Tensor,
                        sensitive_attrs: List[Tuple[str, str]]) -> Dict[str, Dict[str, List]]:
    """
    Organize batch data by demographic groups.

    Args:
        predictions: Tensor of binary predictions [batch_size]
        labels: Tensor of true labels [batch_size]
        sensitive_attrs: List of (left_attr, right_attr) tuples

    Returns:
        Dictionary mapping demographic -> {"preds": [...], "labels": [...]}

    Note:
        For entity matching, we consider the pair's demographic characteristic.
        If both entities are same demographic, use that. If different, mark as "mixed".
    """
    groups = defaultdict(lambda: {"preds": [], "labels": []})

    for i, (left_attr, right_attr) in enumerate(sensitive_attrs):
        # Determine the demographic group for this pair
        if left_attr == right_attr and left_attr is not None:
            group_key = left_attr
        elif left_attr is None or right_attr is None:
            group_key = "unknown"
        else:
            group_key = "mixed"  # Different demographics

        groups[group_key]["preds"].append(predictions[i].item())
        groups[group_key]["labels"].append(labels[i].item())

    return dict(groups)


def calculate_ppvp_for_group(predictions: List[float], labels: List[float],
                             threshold: float = 0.5) -> float:
    """
    Calculate Positive Predictive Value (Precision) for a single demographic group.

    PPVP = TP / (TP + FP)

    Args:
        predictions: List of prediction probabilities
        labels: List of true labels (0 or 1)
        threshold: Classification threshold (default: 0.5)

    Returns:
        PPVP value (0.0 to 1.0), or 0.0 if no positive predictions
    """
    if len(predictions) == 0:
        return 0.0

    # Convert to binary predictions
    binary_preds = [1 if p > threshold else 0 for p in predictions]

    # Calculate TP and FP
    tp = sum(1 for p, l in zip(binary_preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(binary_preds, labels) if p == 1 and l == 0)

    # PPVP = Precision
    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def calculate_ppvp_disparity(predictions: torch.Tensor, labels: torch.Tensor,
                             sensitive_attrs: List[Tuple[str, str]],
                             threshold: float = 0.5) -> float:
    """
    Calculate PPVP disparity across demographic groups within a batch.

    Disparity = max(PPVP) - min(PPVP) across all demographic groups
    Lower disparity = more fair

    Args:
        predictions: Tensor of prediction probabilities [batch_size]
        labels: Tensor of true labels [batch_size]
        sensitive_attrs: List of (left_attr, right_attr) tuples
        threshold: Classification threshold

    Returns:
        Disparity value (0.0 = perfect fairness, higher = more unfair)
    """
    # Group data by demographics
    groups = group_by_demographic(predictions, labels, sensitive_attrs)

    # Calculate PPVP for each group (excluding 'unknown' and 'mixed' for fairness calculation)
    ppvp_values = []
    for group_name, group_data in groups.items():
        if group_name not in ["unknown", "mixed"] and len(group_data["preds"]) > 0:
            ppvp = calculate_ppvp_for_group(group_data["preds"], group_data["labels"], threshold)
            ppvp_values.append(ppvp)

    # If less than 2 groups, no disparity to calculate
    if len(ppvp_values) < 2:
        return 0.0

    # Disparity = max - min
    disparity = max(ppvp_values) - min(ppvp_values)
    return disparity


def calculate_fairness_loss(logits: torch.Tensor, labels: torch.Tensor,
                            sensitive_attrs: List[Tuple[str, str]],
                            fairness_metric: str = "ppvp") -> torch.Tensor:
    """
    Calculate differentiable fairness loss for backpropagation.

    This is the main fairness loss function used during training.
    It converts logits to predictions and calculates the fairness penalty.

    Args:
        logits: Model output logits [batch_size, 2]
        labels: True labels [batch_size]
        sensitive_attrs: List of (left_attr, right_attr) tuples
        fairness_metric: Type of fairness metric ("ppvp" or "tprp") - currently only PPVP implemented

    Returns:
        Differentiable tensor representing fairness penalty

    Note:
        For gradient flow, we use softmax probabilities for the disparity calculation.
        The disparity itself is computed but detached, then multiplied by a differentiable
        component to enable backpropagation.
        The fairness_metric parameter is reserved for future extensibility (TPRP support).
    """
    # Convert logits to probabilities for class 1 (match)
    probs = torch.softmax(logits, dim=1)[:, 1]  # [batch_size]

    # Currently only PPVP is implemented; fairness_metric reserved for future use
    # Calculate PPVP disparity using probabilities
    # Note: We use probs here (differentiable) instead of hard predictions
    disparity = calculate_ppvp_disparity_differentiable(probs, labels, sensitive_attrs)

    return disparity


def calculate_fairness_loss_with_ema(logits: torch.Tensor, labels: torch.Tensor,
                                      sensitive_attrs: List[Tuple[str, str]],
                                      ema_tracker: 'FairnessEMATracker') -> torch.Tensor:
    """
    Calculate fairness loss using EMA-accumulated statistics (Phase B).

    This function addresses the batch-level PPVP issue by:
    1. Updating the EMA tracker with current batch statistics
    2. Calculating disparity from accumulated statistics (stable)
    3. Returning a differentiable loss for gradient flow

    The key insight: We use EMA disparity as the loss VALUE, but multiply by
    a differentiable component (mean of probs) to enable gradient flow.
    This ensures gradients flow back while using stable disparity estimates.

    Args:
        logits: Model output logits [batch_size, 2]
        labels: True labels [batch_size]
        sensitive_attrs: List of (left_attr, right_attr) tuples
        ema_tracker: FairnessEMATracker instance for accumulated statistics

    Returns:
        Differentiable tensor representing fairness penalty
    """
    # Convert logits to probabilities for class 1 (match)
    probs = torch.softmax(logits, dim=1)[:, 1]  # [batch_size]

    # Update EMA tracker with current batch (non-differentiable update)
    ema_tracker.update(probs, labels, sensitive_attrs)

    # Get stable Demographic Parity disparity from accumulated statistics
    # DP is used instead of PPVP because DP is non-degenerate in the
    # all-negative-prediction regime (PPVP becomes 0/0 when soft_tp ~ 0)
    ema_disparity = ema_tracker.get_dp_disparity()

    # Calculate batch-level DP disparity for gradient direction
    batch_disparity = calculate_dp_disparity_differentiable(probs, sensitive_attrs)

    # Create differentiable loss combining EMA stability with batch gradients
    if ema_disparity < 1e-8 and batch_disparity.item() < 1e-8:
        # No disparity at all - return differentiable zero
        return probs.sum() * 0.0

    if batch_disparity.item() < 1e-8:
        # EMA shows disparity but batch doesn't - use EMA value with gradient flow
        return torch.tensor(ema_disparity, device=probs.device, dtype=probs.dtype) * (probs.mean() / probs.mean().detach())
    else:
        # Blend batch and EMA disparity for stability
        scale_factor = ema_disparity / (batch_disparity.item() + 1e-8)
        return batch_disparity * min(scale_factor, 2.0)


def calculate_ppvp_disparity_differentiable(probs: torch.Tensor, labels: torch.Tensor,
                                           sensitive_attrs: List[Tuple[str, str]]) -> torch.Tensor:
    """
    Differentiable version of PPVP disparity calculation for gradient flow.

    Instead of using hard predictions, we use soft probabilities to maintain
    gradient flow during backpropagation.

    Args:
        probs: Prediction probabilities for class 1 [batch_size]
        labels: True labels [batch_size]
        sensitive_attrs: List of (left_attr, right_attr) tuples

    Returns:
        Differentiable disparity tensor
    """
    # Group indices by demographics
    group_indices = defaultdict(list)

    for i, (left_attr, right_attr) in enumerate(sensitive_attrs):
        if left_attr == right_attr and left_attr is not None:
            group_key = left_attr
        else:
            continue  # Skip mixed/unknown for fairness calculation

        group_indices[group_key].append(i)

    # If less than 2 groups, return zero loss (differentiable)
    if len(group_indices) < 2:
        # Return a tensor with gradient by multiplying probs by zero
        return (probs.sum() * 0.0)

    # Calculate soft PPVP for each group
    ppvp_values = []

    for group_name, indices in group_indices.items():
        if len(indices) == 0:
            continue

        # Extract group data
        group_probs = probs[indices]
        group_labels = labels[indices].float()

        # Soft TP: probability * label (element-wise product)
        soft_tp = (group_probs * group_labels).sum()

        # Soft FP: probability * (1 - label)
        soft_fp = (group_probs * (1 - group_labels)).sum()

        # Soft PPVP = soft_TP / (soft_TP + soft_FP)
        soft_ppvp = soft_tp / (soft_tp + soft_fp + 1e-8)  # Add epsilon for stability
        ppvp_values.append(soft_ppvp)

    # Calculate disparity as difference between max and min
    if len(ppvp_values) < 2:
        # Return differentiable zero
        return (probs.sum() * 0.0)

    # Stack and calculate disparity
    ppvp_tensor = torch.stack(ppvp_values)
    disparity = ppvp_tensor.max() - ppvp_tensor.min()

    return disparity


def calculate_dp_disparity_differentiable(probs: torch.Tensor,
                                          sensitive_attrs: List[Tuple[str, str]]) -> torch.Tensor:
    """
    Differentiable Demographic Parity (DP) disparity for gradient flow.

    DP measures the difference in mean prediction probability across groups.
    Unlike PPVP, DP is non-degenerate even when the model predicts all non-matches,
    because it only depends on mean prediction probability, not precision.

    Args:
        probs: Prediction probabilities for class 1 [batch_size]
        sensitive_attrs: List of (left_attr, right_attr) tuples

    Returns:
        Differentiable disparity tensor
    """
    group_indices = defaultdict(list)

    for i, (left_attr, right_attr) in enumerate(sensitive_attrs):
        if left_attr == right_attr and left_attr is not None:
            group_indices[left_attr].append(i)

    if len(group_indices) < 2:
        return probs.sum() * 0.0

    mean_probs = []
    for group_name, indices in group_indices.items():
        if len(indices) == 0:
            continue
        mean_probs.append(probs[indices].mean())

    if len(mean_probs) < 2:
        return probs.sum() * 0.0

    mean_prob_tensor = torch.stack(mean_probs)
    return mean_prob_tensor.max() - mean_prob_tensor.min()


def evaluate_fairness_metrics(predictions: List[float], labels: List[int],
                              sensitive_attrs: List[Tuple[str, str]],
                              threshold: float = 0.5) -> Dict[str, float]:
    """
    Comprehensive fairness evaluation for a complete dataset.

    Args:
        predictions: List of prediction probabilities
        labels: List of true labels
        sensitive_attrs: List of (left_attr, right_attr) tuples
        threshold: Classification threshold

    Returns:
        Dictionary containing:
        - ppvp_disparity: Overall PPVP disparity
        - ppvp_by_group: PPVP value for each demographic group
        - group_counts: Number of samples per group
        - fair_groups: Groups that meet fairness threshold (disparity < 0.1)
    """
    # Convert to tensors for consistency
    pred_tensor = torch.tensor(predictions)
    label_tensor = torch.tensor(labels)

    # Calculate overall disparity
    disparity = calculate_ppvp_disparity(pred_tensor, label_tensor, sensitive_attrs, threshold)

    # Calculate per-group PPVP
    groups = group_by_demographic(pred_tensor, label_tensor, sensitive_attrs)
    ppvp_by_group = {}
    group_counts = {}

    for group_name, group_data in groups.items():
        if len(group_data["preds"]) > 0:
            ppvp = calculate_ppvp_for_group(group_data["preds"], group_data["labels"], threshold)
            ppvp_by_group[group_name] = ppvp
            group_counts[group_name] = len(group_data["preds"])

    # Identify fair groups (those within threshold of each other)
    fair_threshold = 0.1  # Standard fairness threshold from Phase 1
    fair_groups = []

    # Filter to only demographic groups (exclude "unknown" and "mixed")
    demographic_ppvp = {g: v for g, v in ppvp_by_group.items() if g not in ["unknown", "mixed"]}
    total_demographic_groups = len(demographic_ppvp)

    if len(demographic_ppvp) >= 2:
        ppvp_values = list(demographic_ppvp.values())
        max_ppvp = max(ppvp_values)
        min_ppvp = min(ppvp_values)
        actual_disparity = max_ppvp - min_ppvp

        # A group is "fair" only if the overall disparity is below threshold
        # When disparity is below threshold, all groups are fair
        # When disparity is above threshold, only the group(s) closest to the mean are fair
        if actual_disparity <= fair_threshold:
            fair_groups = list(demographic_ppvp.keys())
        else:
            avg_ppvp = sum(ppvp_values) / len(ppvp_values)
            for group_name, ppvp in demographic_ppvp.items():
                if abs(ppvp - avg_ppvp) <= fair_threshold / 2:
                    fair_groups.append(group_name)

    # Fairness rate: based on actual disparity vs threshold, not group counting
    # This prevents the rate from being vacuously 1.0 when both groups have PPVP=0
    if total_demographic_groups >= 2:
        ppvp_vals = [v for g, v in ppvp_by_group.items() if g not in ["unknown", "mixed"]]
        has_any_positive_preds = any(v > 0 for v in ppvp_vals)
        if not has_any_positive_preds:
            # Both groups have PPVP=0 → model makes no positive predictions → fairness is undefined
            actual_fairness_rate = 0.0  # Not vacuously 1.0
        else:
            actual_fairness_rate = 1.0 if disparity <= fair_threshold else 0.0
    else:
        actual_fairness_rate = 0.0  # Can't assess fairness with <2 groups

    return {
        'ppvp_disparity': disparity,
        'ppvp_by_group': ppvp_by_group,
        'group_counts': group_counts,
        'fair_groups': fair_groups,
        'num_fair_groups': len(fair_groups),
        'total_groups': total_demographic_groups,
        'fairness_rate': actual_fairness_rate
    }


if __name__ == "__main__":
    # Simple validation tests
    print("Testing fairness_utils.py")

    # Test 1: Extract sensitive attribute
    test_text = "COL id VAL 123 COL FullName VAL John Smith COL Ethnic_Code_Text VAL African-American"
    attr = extract_sensitive_attribute(test_text)
    print(f"[PASS] Extract attribute: {attr}")
    assert attr == "African-American", f"Expected 'African-American', got '{attr}'"

    # Test 2: PPVP calculation
    preds = [0.9, 0.8, 0.3, 0.2]  # 2 positive, 2 negative predictions
    labels = [1, 1, 0, 0]  # 2 matches, 2 non-matches
    ppvp = calculate_ppvp_for_group(preds, labels)
    print(f"[PASS] PPVP calculation: {ppvp:.3f}")
    assert ppvp == 1.0, f"Expected 1.0 (perfect precision), got {ppvp}"

    # Test 3: Differentiable loss
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]], requires_grad=True)  # 2 samples, 2 classes
    labels = torch.tensor([1, 0])
    attrs = [("African-American", "African-American"), ("Caucasian", "Caucasian")]
    loss = calculate_fairness_loss(logits, labels, attrs)
    print(f"[PASS] Fairness loss: {loss.item():.4f} (requires_grad={loss.requires_grad})")
    assert loss.requires_grad, "Loss should be differentiable"

    # Test gradient flow
    loss.backward()
    assert logits.grad is not None, "Gradients should flow to logits"
    print(f"[PASS] Gradient flow verified (grad shape: {logits.grad.shape})")

    print("\n[SUCCESS] All validation tests passed!")
