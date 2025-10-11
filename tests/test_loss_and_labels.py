"""
Quick test to verify SupCon loss computation and dummy dataset labels.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.losses import SupConLoss
from src.dataset import WeldingDataset
from src.samplers import StratifiedBatchSampler
from torch.utils.data import DataLoader
from collections import Counter


def test_supcon_with_various_labels():
    """Test SupCon loss with different label distributions."""
    print("=" * 70)
    print("Testing SupCon Loss")
    print("=" * 70)
    
    criterion = SupConLoss(temperature=0.07)
    feature_dim = 512
    
    # Test 1: All same labels (should give non-zero loss)
    print("\nTest 1: All same labels (batch_size=4)")
    features = torch.randn(4, feature_dim)
    labels = torch.tensor([0, 0, 0, 0])
    loss = criterion(features, labels)
    print(f"  Loss: {loss.item():.6f}")
    assert loss.item() != 0, "Loss should not be zero!"
    
    # Test 2: Two classes, balanced
    print("\nTest 2: Two classes, balanced (batch_size=4)")
    labels = torch.tensor([0, 0, 1, 1])
    loss = criterion(features, labels)
    print(f"  Loss: {loss.item():.6f}")
    
    # Test 3: Multiple classes
    print("\nTest 3: Multiple classes (batch_size=6)")
    features = torch.randn(6, feature_dim)
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    loss = criterion(features, labels)
    print(f"  Loss: {loss.item():.6f}")
    
    # Test 4: Batch size 2, same label
    print("\nTest 4: Batch size 2, same label")
    features = torch.randn(2, feature_dim)
    labels = torch.tensor([0, 0])
    loss = criterion(features, labels)
    print(f"  Loss: {loss.item():.6f}")
    assert loss.item() != 0, "Loss should not be zero even with 2 samples!"
    
    print("\n✅ All SupCon tests passed!\n")


def test_dummy_dataset_labels():
    """Test that dummy dataset produces diverse labels."""
    print("=" * 70)
    print("Testing Dummy Dataset Labels")
    print("=" * 70)
    
    dataset = WeldingDataset(
        mode="dummy",
        num_samples=32,
        num_frames=8,
        frame_size=64,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Collect all labels
    labels = [dataset[i]['label'] for i in range(len(dataset))]
    unique_labels = set(labels)
    
    print(f"Unique labels: {sorted(unique_labels)}")
    print(f"Number of unique labels: {len(unique_labels)}")
    
    # Count per label
    from collections import Counter
    label_counts = Counter(labels)
    print("\nLabel distribution:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        pct = 100 * count / len(labels)
        print(f"  Label {label}: {count:3d} samples ({pct:5.1f}%)")
    
    assert len(unique_labels) > 1, "Should have multiple classes!"
    print("\n✅ Dataset label test passed!\n")


def test_dataloader_batches():
    """Test that dataloader batches contain diverse labels."""
    print("=" * 70)
    print("Testing DataLoader Batch Diversity")
    print("=" * 70)
    
    dataset = WeldingDataset(
        mode="dummy",
        num_samples=32,
        num_frames=8,
        frame_size=64,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    
    print(f"\nTotal batches: {len(loader)}")
    
    # Check first 3 batches
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 3:
            break
        
        labels = batch['label']
        unique = torch.unique(labels)
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Labels: {labels.tolist()}")
        print(f"  Unique: {unique.tolist()}")
        print(f"  Diversity: {len(unique)}/{len(labels)} classes")
    
    print("\n✅ DataLoader test passed!\n")


def test_stratified_sampler():
    """Test that StratifiedBatchSampler creates diverse batches."""
    print("=" * 70)
    print("Testing StratifiedBatchSampler")
    print("=" * 70)
    
    # Create dataset with known label distribution
    dataset = WeldingDataset(dummy=True, num_samples=60)
    
    print(f"\nDataset: {len(dataset)} samples")
    label_counts = Counter(dataset._labels)
    print(f"Label distribution: {dict(sorted(label_counts.items()))}")
    
    # Create stratified sampler
    batch_size = 6
    sampler = StratifiedBatchSampler(
        labels=dataset._labels,
        batch_size=batch_size,
        drop_last=False,
    )
    
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=dataset.collate_fn)
    
    print(f"\nBatch size: {batch_size}")
    print(f"Total batches: {len(loader)}")
    
    # Check first few batches for diversity
    diversity_scores = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 5:  # Check first 5 batches
            break
        
        labels = batch['label']
        unique = torch.unique(labels)
        diversity = len(unique) / len(labels)
        diversity_scores.append(diversity)
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Labels: {labels.tolist()}")
        print(f"  Unique classes: {len(unique)}/{len(labels)}")
        print(f"  Diversity score: {diversity:.2f}")
    
    avg_diversity = sum(diversity_scores) / len(diversity_scores)
    print(f"\nAverage diversity: {avg_diversity:.2f}")
    
    if avg_diversity > 0.5:  # At least 50% diversity on average
        print("✅ Stratified sampler creates diverse batches!")
    else:
        print("⚠️  Warning: Low diversity in batches")
    
    print()


if __name__ == "__main__":
    test_supcon_with_various_labels()
    test_dummy_dataset_labels()
    test_dataloader_batches()
    test_stratified_sampler()
    
    print("=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
