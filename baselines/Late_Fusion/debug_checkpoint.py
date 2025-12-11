"""Debug script to inspect checkpoint contents."""

import torch
import sys

if len(sys.argv) < 2:
    print("Usage: python debug_checkpoint.py <checkpoint_path>")
    sys.exit(1)

checkpoint_path = sys.argv[1]
print(f"Loading checkpoint: {checkpoint_path}")

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n" + "="*70)
    print("CHECKPOINT CONTENTS")
    print("="*70)
    
    print("\nKeys in checkpoint:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    if 'config' in checkpoint:
        print("\nSaved config:")
        config = checkpoint['config']
        for k, v in config.items():
            print(f"  {k}: {v}")
        
        if 'n_fft' in config:
            n_bins = config['n_fft'] // 2 + 1
            print(f"\nCalculated n_bins: {n_bins}")
    
    if 'model_state_dict' in checkpoint:
        print("\nModel state_dict keys (first 10):")
        state_dict = checkpoint['model_state_dict']
        for i, key in enumerate(list(state_dict.keys())[:10]):
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
            print(f"  {key}: {shape}")
        
        # Check BatchNorm running_mean
        if 'encoder.0.running_mean' in state_dict:
            running_mean = state_dict['encoder.0.running_mean']
            print(f"\nBatchNorm running_mean shape: {running_mean.shape}")
            print(f"BatchNorm running_mean elements: {running_mean.numel()}")
        
        # Check first conv layer
        if 'encoder.1.weight' in state_dict:
            conv_weight = state_dict['encoder.1.weight']
            print(f"\nFirst Conv1d weight shape: {conv_weight.shape}")
            print(f"  in_channels: {conv_weight.shape[1]}")
            print(f"  out_channels: {conv_weight.shape[0]}")
    
    print("\n" + "="*70)
    
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()
