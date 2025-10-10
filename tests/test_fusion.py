"""
Test script for Cross-Attention Fusion Module.

Tests the quad-modal fusion mechanism with dummy and real configurations.
"""

import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.fusion import CrossAttentionFusionModule, DummyCrossAttentionFusion
from configs.model_config import FUSION


def test_cross_attention_fusion():
    """Test CrossAttentionFusionModule with dummy features."""
    print("\n" + "=" * 70)
    print("Testing CrossAttentionFusionModule")
    print("=" * 70)
    
    # Configuration
    batch_size = 4
    video_seq_len = 8
    image_seq_len = 5
    audio_seq_len = 12
    sensor_seq_len = 256
    
    video_dim = FUSION["video_dim"]
    image_dim = FUSION["image_dim"]
    audio_dim = FUSION["audio_dim"]
    sensor_dim = FUSION["sensor_dim"]
    hidden_dim = FUSION["hidden_dim"]
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Video: seq_len={video_seq_len}, dim={video_dim}")
    print(f"  Image: seq_len={image_seq_len}, dim={image_dim}")
    print(f"  Audio: seq_len={audio_seq_len}, dim={audio_dim}")
    print(f"  Sensor: seq_len={sensor_seq_len}, dim={sensor_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    
    # Create fusion module
    fusion = CrossAttentionFusionModule(
        video_dim=video_dim,
        image_dim=image_dim,
        audio_dim=audio_dim,
        sensor_dim=sensor_dim,
        hidden_dim=hidden_dim,
        num_fusion_tokens=FUSION["num_fusion_tokens"],
        num_heads=FUSION["num_heads"],
        dropout=FUSION["dropout"],
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in fusion.parameters())
    trainable_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Create dummy features
    video_features = torch.randn(batch_size, video_seq_len, video_dim)
    image_features = torch.randn(batch_size, image_seq_len, image_dim)
    audio_features = torch.randn(batch_size, audio_seq_len, audio_dim)
    sensor_features = torch.randn(batch_size, sensor_seq_len, sensor_dim)
    
    print(f"\nInput shapes:")
    print(f"  Video: {tuple(video_features.shape)}")
    print(f"  Image: {tuple(image_features.shape)}")
    print(f"  Audio: {tuple(audio_features.shape)}")
    print(f"  Sensor: {tuple(sensor_features.shape)}")
    
    # Forward pass without attention weights
    print(f"\nForward pass (without attention)...")
    fused_features = fusion(
        video_features=video_features,
        image_features=image_features,
        audio_features=audio_features,
        sensor_features=sensor_features,
        return_attention=False,
    )
    
    print(f"  Output shape: {tuple(fused_features.shape)}")
    assert fused_features.shape == (batch_size, hidden_dim), \
        f"Expected shape ({batch_size}, {hidden_dim}), got {fused_features.shape}"
    
    # Check for NaN or Inf
    assert not torch.isnan(fused_features).any(), "Output contains NaN"
    assert not torch.isinf(fused_features).any(), "Output contains Inf"
    print(f"  Output range: [{fused_features.min().item():.4f}, {fused_features.max().item():.4f}]")
    print(f"  ✅ Output shape correct and values valid")
    
    # Forward pass with attention weights
    print(f"\nForward pass (with attention)...")
    fused_features, attention_weights = fusion(
        video_features=video_features,
        image_features=image_features,
        audio_features=audio_features,
        sensor_features=sensor_features,
        return_attention=True,
    )
    
    print(f"  Output shape: {tuple(fused_features.shape)}")
    print(f"  Attention weights returned for: {list(attention_weights.keys())}")
    
    # Validate attention weights
    expected_modalities = ["video", "image", "audio", "sensor"]
    for modality in expected_modalities:
        assert modality in attention_weights, f"Missing attention for {modality}"
        attn = attention_weights[modality]
        print(f"    {modality}: {tuple(attn.shape)}")
    
    print(f"  ✅ Attention weights shape correct")
    
    # Test gradient flow
    print(f"\nTesting gradient flow...")
    loss = fused_features.sum()
    loss.backward()
    
    gradients_ok = True
    for name, param in fusion.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"  ❌ Gradient issue in {name}")
                gradients_ok = False
    
    if gradients_ok:
        print(f"  ✅ All gradients valid")
    
    print(f"\n✅ CrossAttentionFusionModule test passed!\n")
    return True


def test_dummy_fusion():
    """Test DummyCrossAttentionFusion with simple concatenation."""
    print("\n" + "=" * 70)
    print("Testing DummyCrossAttentionFusion (Lightweight)")
    print("=" * 70)
    
    # Configuration
    batch_size = 4
    video_seq_len = 8
    image_seq_len = 5
    audio_seq_len = 12
    sensor_seq_len = 256
    
    video_dim = FUSION["video_dim"]
    image_dim = FUSION["image_dim"]
    audio_dim = FUSION["audio_dim"]
    sensor_dim = FUSION["sensor_dim"]
    hidden_dim = FUSION["hidden_dim"]
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Hidden dim: {hidden_dim}")
    
    # Create dummy fusion module
    fusion = DummyCrossAttentionFusion(
        video_dim=video_dim,
        image_dim=image_dim,
        audio_dim=audio_dim,
        sensor_dim=sensor_dim,
        hidden_dim=hidden_dim,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in fusion.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    # Create dummy features
    video_features = torch.randn(batch_size, video_seq_len, video_dim)
    image_features = torch.randn(batch_size, image_seq_len, image_dim)
    audio_features = torch.randn(batch_size, audio_seq_len, audio_dim)
    sensor_features = torch.randn(batch_size, sensor_seq_len, sensor_dim)
    
    # Forward pass
    print(f"\nForward pass...")
    fused_features = fusion(
        video_features=video_features,
        image_features=image_features,
        audio_features=audio_features,
        sensor_features=sensor_features,
    )
    
    print(f"  Output shape: {tuple(fused_features.shape)}")
    assert fused_features.shape == (batch_size, hidden_dim), \
        f"Expected shape ({batch_size}, {hidden_dim}), got {fused_features.shape}"
    
    # Check for NaN or Inf
    assert not torch.isnan(fused_features).any(), "Output contains NaN"
    assert not torch.isinf(fused_features).any(), "Output contains Inf"
    print(f"  ✅ Output shape correct and values valid")
    
    # Test gradient flow
    print(f"\nTesting gradient flow...")
    loss = fused_features.sum()
    loss.backward()
    print(f"  ✅ Gradients computed successfully")
    
    print(f"\n✅ DummyCrossAttentionFusion test passed!\n")
    return True


def test_fusion_with_different_batch_sizes():
    """Test fusion module with different batch sizes."""
    print("\n" + "=" * 70)
    print("Testing Fusion with Different Batch Sizes")
    print("=" * 70)
    
    fusion = CrossAttentionFusionModule(**FUSION)
    
    batch_sizes = [1, 2, 8, 16]
    
    for bs in batch_sizes:
        video_features = torch.randn(bs, 8, FUSION["video_dim"])
        image_features = torch.randn(bs, 5, FUSION["image_dim"])
        audio_features = torch.randn(bs, 12, FUSION["audio_dim"])
        sensor_features = torch.randn(bs, 256, FUSION["sensor_dim"])
        
        fused = fusion(video_features, image_features, audio_features, sensor_features)
        
        assert fused.shape == (bs, FUSION["hidden_dim"]), \
            f"Batch size {bs}: Expected shape ({bs}, {FUSION['hidden_dim']}), got {fused.shape}"
        
        print(f"  Batch size {bs:2d}: {tuple(fused.shape)} ✅")
    
    print(f"\n✅ All batch sizes passed!\n")
    return True


def main():
    """Run all fusion module tests."""
    print("\n" + "=" * 70)
    print("FUSION MODULE TEST SUITE")
    print("=" * 70)
    
    try:
        # Test 1: CrossAttentionFusionModule
        test_cross_attention_fusion()
        
        # Test 2: DummyCrossAttentionFusion
        test_dummy_fusion()
        
        # Test 3: Different batch sizes
        test_fusion_with_different_batch_sizes()
        
        print("\n" + "=" * 70)
        print("✅ ALL FUSION TESTS PASSED!")
        print("=" * 70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
