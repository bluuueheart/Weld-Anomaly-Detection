#!/usr/bin/env python
"""Test dataset loading functionality."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "configs"))

from dataset import WeldingDataset, collate_fn
from dataset_config import *


def test_dataset():
    """Test dataset loading."""
    print("="*80)
    print("Testing WeldingDataset - Real Mode")
    print("="*80)
    
    dataset = WeldingDataset(
        root_dir=str(project_root / DATA_ROOT),
        mode="real",
        num_frames=VIDEO_NUM_FRAMES,
        frame_size=VIDEO_FRAME_SIZE,
        image_size=IMAGE_SIZE,
        num_angles=IMAGE_NUM_ANGLES,
        audio_mel_bins=AUDIO_N_MELS,
        audio_frames=AUDIO_FRAMES,
        audio_sr=AUDIO_SAMPLE_RATE,
        sensor_len=SENSOR_LEN,
        sensor_channels=SENSOR_CHANNELS,
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    
    # Label distribution
    label_counts = {}
    for label in dataset._labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\nLabel Distribution:")
    cat_names = {v: k for k, v in CATEGORIES.items()}
    for label, count in sorted(label_counts.items()):
        print(f"  {label} ({cat_names.get(label, 'unknown')}): {count}")
    
    if len(dataset) == 0:
        print("\nNo samples found!")
        return
    
    # Test loading first sample
    print(f"\n{'='*80}")
    print("Testing Sample Loading - Index 0")
    print("="*80)
    
    try:
        sample = dataset[0]
        print(f"\nSample loaded successfully!")
        print(f"  ID: {sample['meta']['id']}")
        print(f"  Label: {sample['label']}")
        print(f"  Video shape: {sample['video'].shape}")
        print(f"  Post-weld images shape: {sample['post_weld_images'].shape}")
        print(f"  Audio shape: {sample['audio'].shape}")
        print(f"  Sensor shape: {sample['sensor'].shape}")
        
        # Check data types and ranges
        print(f"\nData Quality:")
        print(f"  Video: {sample['video'].dtype}, [{sample['video'].min():.3f}, {sample['video'].max():.3f}]")
        print(f"  Images: {sample['post_weld_images'].dtype}, [{sample['post_weld_images'].min():.3f}, {sample['post_weld_images'].max():.3f}]")
        print(f"  Audio: {sample['audio'].dtype}, [{sample['audio'].min():.3f}, {sample['audio'].max():.3f}]")
        print(f"  Sensor: {sample['sensor'].dtype}, [{sample['sensor'].min():.3f}, {sample['sensor'].max():.3f}]")
        
    except Exception as e:
        print(f"\nError loading sample: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test batch loading
    if len(dataset) >= 4:
        print(f"\n{'='*80}")
        print("Testing Batch Loading")
        print("="*80)
        
        try:
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=4, shuffle=False, 
                              collate_fn=collate_fn, num_workers=0)
            batch = next(iter(loader))
            print(f"\nBatch loaded successfully!")
            print(f"  Video batch: {batch['video'].shape}")
            print(f"  Post-weld images batch: {batch['post_weld_images'].shape}")
            print(f"  Audio batch: {batch['audio'].shape}")
            print(f"  Sensor batch: {batch['sensor'].shape}")
            print(f"  Labels batch: {batch['label'].shape}")
        except Exception as e:
            print(f"\nError loading batch: {e}")
    
    print(f"\n{'='*80}")
    print("Test Completed!")
    print("="*80)


if __name__ == "__main__":
    test_dataset()
