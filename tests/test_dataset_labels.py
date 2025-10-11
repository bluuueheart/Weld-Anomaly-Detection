"""
Debug script to check dataset label distribution.
Run this before training to verify labels are correctly parsed.
"""
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import WeldingDataset
from configs.dataset_config import *

def main():
    print("=" * 70)
    print("Dataset Label Distribution Check")
    print("=" * 70)
    print()
    
    # Load real dataset
    dataset = WeldingDataset(
        data_root=DATA_ROOT,
        video_length=VIDEO_LENGTH,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        audio_duration=AUDIO_DURATION,
        sensor_length=SENSOR_LENGTH,
        image_size=IMAGE_SIZE,
        num_angles=IMAGE_NUM_ANGLES,
        dummy=False,  # Use real data
    )
    
    print(f"Total samples: {len(dataset)}")
    print()
    
    # Check label distribution
    labels = [dataset._labels[i] for i in range(len(dataset))]
    label_counts = Counter(labels)
    
    # Category names
    cat_names = {
        0: "good_weld",
        1: "crater_cracks",
        2: "burn_through",
        3: "excessive_penetration",
        4: "porosity",
        5: "spatter",
    }
    
    print("Label Distribution:")
    print("-" * 70)
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        name = cat_names.get(label, f"unknown_{label}")
        pct = 100.0 * count / len(dataset)
        print(f"  [{label}] {name:25s}: {count:4d} ({pct:5.1f}%)")
    
    print()
    
    # Show first 10 samples
    print("First 10 samples:")
    print("-" * 70)
    for i in range(min(10, len(dataset))):
        sid = dataset._ids[i]
        label = dataset._labels[i]
        name = cat_names.get(label, f"unknown_{label}")
        print(f"  {i:3d}: [{label}] {name:25s} <- {sid}")
    
    print()
    
    # Warning if imbalanced
    unique_labels = len(label_counts)
    if unique_labels < 2:
        print("⚠️  WARNING: Only 1 class found! SupCon loss will not work.")
        print("   Check your Data/ directory structure.")
        return False
    elif unique_labels < 6:
        print(f"⚠️  WARNING: Only {unique_labels}/6 classes found.")
        print("   Some categories may be missing.")
    
    # Check if extremely imbalanced
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    if max_count / min_count > 10:
        print(f"⚠️  WARNING: Very imbalanced dataset (max/min = {max_count/min_count:.1f})")
        print("   Consider using weighted sampling or data augmentation.")
    
    print()
    print("✅ Label distribution check complete!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
