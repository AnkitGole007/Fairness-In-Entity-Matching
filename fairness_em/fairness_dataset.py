"""
Fairness-Aware Dataset for Phase 3: Loss Curve Exploration

This module extends Ditto's DittoDataset to extract and return sensitive attributes
alongside the standard input data, enabling fairness-aware training.

Key Features:
- Extracts demographic information from COL/VAL format
- Maintains compatibility with existing Ditto training pipeline
- Returns enhanced batches with sensitive attributes
"""

import sys
import os
import torch

# Add ditto_light to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ditto'))

from ditto_light.dataset import DittoDataset, get_tokenizer
from typing import List, Tuple, Optional, Union
from fairness_utils import extract_sensitive_attributes_from_pair


class FairnessDittoDataset(DittoDataset):
    """
    Enhanced DittoDataset that extracts and returns sensitive attributes.

    This dataset extends the original DittoDataset by:
    1. Parsing entity pairs to extract sensitive attributes (e.g., Ethnic_Code_Text)
    2. Storing sensitive attributes alongside pairs and labels
    3. Returning enhanced batches that include demographic information

    The dataset maintains full compatibility with Ditto's training pipeline while
    enabling fairness-aware loss calculations.
    """

    def __init__(self,
                 path: Union[str, List[str]],
                 max_len: int = 256,
                 size: Optional[int] = None,
                 lm: str = 'roberta',
                 da: Optional[str] = None,
                 sensitive_attribute: str = 'Ethnic_Code_Text'):
        """
        Initialize FairnessDittoDataset.

        Args:
            path: Path to data file or list of lines
            max_len: Maximum sequence length for tokenization
            size: Optional dataset size limit
            lm: Language model name ('roberta', 'distilbert', etc.)
            da: Data augmentation type (None, 'del', 'swap', etc.)
            sensitive_attribute: Name of the sensitive attribute to extract
        """
        # Initialize parent DittoDataset
        super().__init__(path, max_len, size, lm, da)

        # Store sensitive attribute configuration
        self.sensitive_attribute = sensitive_attribute
        self.sensitive_attrs = []

        # Extract sensitive attributes from all pairs
        self._extract_sensitive_attributes()

    def _extract_sensitive_attributes(self):
        """
        Extract sensitive attributes from stored entity pairs.

        Process:
        1. For each (left, right) pair in self.pairs
        2. Extract sensitive attribute from both entities
        3. Store as tuple: (left_attr, right_attr)

        Note:
            If extraction fails, stores (None, None) to maintain alignment
        """
        for left_text, right_text in self.pairs:
            # Extract sensitive attributes from both entities
            left_attr, right_attr = extract_sensitive_attributes_from_pair(
                left_text,
                right_text,
                self.sensitive_attribute
            )
            self.sensitive_attrs.append((left_attr, right_attr))

        # Verify alignment
        assert len(self.sensitive_attrs) == len(self.pairs), \
            "Sensitive attributes list must match pairs list"

    def __getitem__(self, idx: int):
        """
        Return enhanced batch item with sensitive attributes.

        Args:
            idx: Index of the item to retrieve

        Returns:
            Without data augmentation: (input_ids, label, sensitive_attrs)
            With data augmentation: (input_ids_1, input_ids_2, label, sensitive_attrs)

        The sensitive_attrs is a tuple: (left_attribute, right_attribute)
        """
        left = self.pairs[idx][0]
        right = self.pairs[idx][1]
        label = self.labels[idx]
        sensitive_attr = self.sensitive_attrs[idx]

        # Tokenize left + right
        x = self.tokenizer.encode(text=left,
                                  text_pair=right,
                                  max_length=self.max_len,
                                  truncation=True)

        # Return with sensitive attributes
        if self.da is not None:
            # Data augmentation enabled
            combined = self.augmenter.augment_sent(left + ' [SEP] ' + right, self.da)
            left_aug, right_aug = combined.split(' [SEP] ')
            x_aug = self.tokenizer.encode(text=left_aug,
                                      text_pair=right_aug,
                                      max_length=self.max_len,
                                      truncation=True)
            return x, x_aug, label, sensitive_attr
        else:
            # No data augmentation
            return x, label, sensitive_attr

    @staticmethod
    def pad(batch):
        """
        Enhanced padding function that handles sensitive attributes.

        Args:
            batch: List of tuples from __getitem__
                   Format: [(x, label, sensitive_attr), ...] or
                          [(x1, x2, label, sensitive_attr), ...]

        Returns:
            Without DA: (padded_x, labels_tensor, sensitive_attrs_list)
            With DA: (padded_x1, padded_x2, labels_tensor, sensitive_attrs_list)

        The sensitive attributes are returned as a list (not tensor) since
        they are categorical strings used for grouping during loss calculation.
        """
        # Check if data augmentation is enabled (4-tuple vs 3-tuple)
        if len(batch[0]) == 4:
            # With data augmentation: (x1, x2, label, sensitive_attr)
            x1, x2, y, sensitive_attrs = zip(*batch)

            # Find max length and pad
            maxlen = max([len(x) for x in x1+x2])
            x1_padded = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            x2_padded = [xi + [0]*(maxlen - len(xi)) for xi in x2]

            return (torch.LongTensor(x1_padded),
                   torch.LongTensor(x2_padded),
                   torch.LongTensor(y),
                   list(sensitive_attrs))
        else:
            # Without data augmentation: (x, label, sensitive_attr)
            x, y, sensitive_attrs = zip(*batch)

            # Find max length and pad
            maxlen = max([len(xi) for xi in x])
            x_padded = [xi + [0]*(maxlen - len(xi)) for xi in x]

            return (torch.LongTensor(x_padded),
                   torch.LongTensor(y),
                   list(sensitive_attrs))

    def get_sensitive_attribute_stats(self) -> dict:
        """
        Get statistics about sensitive attributes in the dataset.

        Returns:
            Dictionary with:
            - total_samples: Total number of samples
            - same_group: Number of pairs with same demographic
            - mixed_group: Number of pairs with different demographics
            - unknown: Number of pairs with missing attributes
            - group_counts: Distribution of demographic groups
        """
        same_group = 0
        mixed_group = 0
        unknown = 0
        group_distribution = {}

        for left_attr, right_attr in self.sensitive_attrs:
            if left_attr is None or right_attr is None:
                unknown += 1
            elif left_attr == right_attr:
                same_group += 1
                group_distribution[left_attr] = group_distribution.get(left_attr, 0) + 1
            else:
                mixed_group += 1

        return {
            'total_samples': len(self.sensitive_attrs),
            'same_group': same_group,
            'mixed_group': mixed_group,
            'unknown': unknown,
            'group_distribution': group_distribution
        }


if __name__ == "__main__":
    # Validation tests with ACTUAL COMPAS DATASET
    print("Testing fairness_dataset.py with Compas Dataset")
    print("="*60)

    # Path to actual Compas data
    compas_train_path = os.path.join(os.path.dirname(__file__), '..', 'ditto', 'data', 'Compas', 'train.txt')

    try:
        # Test 1: Load actual Compas dataset (limited size for testing)
        print("\n[Test 1] Loading Compas training data...")
        dataset = FairnessDittoDataset(compas_train_path, max_len=128, lm='distilbert', size=100)
        print(f"[PASS] Dataset loaded: {len(dataset)} samples")

        # Test 2: Check sensitive attribute extraction from real data
        print("\n[Test 2] Analyzing sensitive attributes...")
        stats = dataset.get_sensitive_attribute_stats()
        print(f"[PASS] Attribute statistics:")
        print(f"  - Total samples: {stats['total_samples']}")
        print(f"  - Same demographic: {stats['same_group']}")
        print(f"  - Mixed demographics: {stats['mixed_group']}")
        print(f"  - Unknown/missing: {stats['unknown']}")
        print(f"  - Group distribution: {stats['group_distribution']}")

        # Verify attributes were extracted
        assert stats['total_samples'] == 100, "Should have 100 samples"
        assert stats['same_group'] + stats['mixed_group'] + stats['unknown'] == stats['total_samples'], \
            "Categories should sum to total"

        # Test 3: Test __getitem__ with real Compas data
        print("\n[Test 3] Testing data retrieval...")
        item = dataset[0]
        assert len(item) == 3, f"Expected 3-tuple (no DA), got {len(item)}-tuple"
        input_ids, label, sensitive_attr = item
        print(f"[PASS] Sample 0:")
        print(f"  - Input IDs length: {len(input_ids)}")
        print(f"  - Label: {label}")
        print(f"  - Sensitive attributes: {sensitive_attr}")

        # Test 4: Test batching with PyTorch DataLoader
        print("\n[Test 4] Testing DataLoader compatibility...")
        from torch.utils.data import DataLoader

        # Create DataLoader with custom collate function
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=FairnessDittoDataset.pad
        )

        # Get one batch
        for batch in dataloader:
            x_tensor, y_tensor, attrs_list = batch
            print(f"[PASS] DataLoader batch:")
            print(f"  - Input shape: {x_tensor.shape}")
            print(f"  - Labels shape: {y_tensor.shape}")
            print(f"  - Attributes count: {len(attrs_list)}")
            print(f"  - Sample attributes: {attrs_list[:3]}")

            # Verify shapes
            assert x_tensor.shape[0] == 8, "Batch size should be 8"
            assert y_tensor.shape[0] == 8, "Labels batch size should be 8"
            assert len(attrs_list) == 8, "Should have 8 attribute tuples"
            break  # Only test first batch

        # Test 5: Verify no None values in critical data
        print("\n[Test 5] Data integrity check...")
        none_count = sum(1 for left, right in dataset.sensitive_attrs if left is None or right is None)
        print(f"[INFO] Samples with missing attributes: {none_count}/{len(dataset)}")

        print("\n" + "="*60)
        print("[SUCCESS] All Compas dataset validation tests passed!")
        print("FairnessDittoDataset is ready for Phase 3 training!")

    except FileNotFoundError:
        print(f"\n[ERROR] Compas dataset not found at: {compas_train_path}")
        print("Please ensure the Compas dataset is properly placed in ditto/data/Compas/")
        raise
    except Exception as e:
        print(f"\n[FAILED] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
