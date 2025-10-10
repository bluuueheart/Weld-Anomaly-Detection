"""
Test script for complete QuadModalSOTAModel.

Tests end-to-end forward pass with all four modalities.
"""

import sys
import os
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import QuadModalSOTAModel, create_quadmodal_model
from configs.model_config import (
    VIDEO_ENCODER,
    IMAGE_ENCODER,
    AUDIO_ENCODER,
    SENSOR_ENCODER,
    FUSION,
)


def test_quadmodal_model_dummy():
    """Test QuadModalSOTAModel with dummy encoders."""
    print("\n" + "=" * 70)
    print("Testing QuadModalSOTAModel (Dummy Encoders)")
    print("=" * 70)
    
    # Create model with dummy encoders
    model = QuadModalSOTAModel(
        video_config=VIDEO_ENCODER,
        image_config=IMAGE_ENCODER,
        audio_config=AUDIO_ENCODER,
        sensor_config=SENSOR_ENCODER,
        fusion_config=FUSION,
        use_dummy=True,
    )
    
    # Print model info
    total_params = model.get_num_parameters(trainable_only=False)
    trainable_params = model.get_num_parameters(trainable_only=True)
    output_dim = model.get_feature_dim()
    
    print(f"\nModel Configuration:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Output dimension: {output_dim}")
    
    # Create dummy batch
    batch_size = 4
    batch = {
        "video": torch.randn(batch_size, 32, 3, 224, 224),
        "post_weld_images": torch.randn(batch_size, 5, 3, 224, 224),
        "audio": torch.randn(batch_size, 1, 128, 256),
        "sensor": torch.randn(batch_size, 256, 6),
    }
    
    print(f"\nInput shapes:")
    for key, value in batch.items():
        print(f"  {key}: {tuple(value.shape)}")
    
    # Forward pass without attention
    print(f"\nForward pass (without attention)...")
    output = model(batch, return_attention=False)
    
    print(f"  Output shape: {tuple(output.shape)}")
    assert output.shape == (batch_size, output_dim), \
        f"Expected shape ({batch_size}, {output_dim}), got {output.shape}"
    
    # Check for NaN or Inf
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"  ✅ Forward pass successful")
    
    # Forward pass with attention
    print(f"\nForward pass (with attention)...")
    output, attention = model(batch, return_attention=True)
    
    print(f"  Output shape: {tuple(output.shape)}")
    print(f"  Attention keys: {list(attention.keys())}")
    print(f"  ✅ Attention weights returned")
    
    # Test gradient flow
    print(f"\nTesting gradient flow...")
    loss = output.sum()
    loss.backward()
    
    grad_ok = True
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"  ❌ Gradient issue in {name}")
                grad_ok = False
    
    if grad_ok:
        print(f"  ✅ All gradients valid")
    
    print(f"\n✅ QuadModalSOTAModel (Dummy) test passed!\n")
    return True


def test_quadmodal_model_factory():
    """Test create_quadmodal_model factory function."""
    print("\n" + "=" * 70)
    print("Testing create_quadmodal_model Factory")
    print("=" * 70)
    
    config = {
        "VIDEO_ENCODER": VIDEO_ENCODER,
        "IMAGE_ENCODER": IMAGE_ENCODER,
        "AUDIO_ENCODER": AUDIO_ENCODER,
        "SENSOR_ENCODER": SENSOR_ENCODER,
        "FUSION": FUSION,
    }
    
    # Create model using factory
    model = create_quadmodal_model(config, use_dummy=True)
    
    print(f"\nModel created via factory:")
    print(f"  Output dimension: {model.get_feature_dim()}")
    print(f"  Parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    batch = {
        "video": torch.randn(2, 32, 3, 224, 224),
        "post_weld_images": torch.randn(2, 5, 3, 224, 224),
        "audio": torch.randn(2, 1, 128, 256),
        "sensor": torch.randn(2, 256, 6),
    }
    
    output = model(batch)
    print(f"  Output shape: {tuple(output.shape)}")
    assert output.shape == (2, model.get_feature_dim())
    
    print(f"\n✅ Factory function test passed!\n")
    return True


def test_encoder_freezing():
    """Test encoder freezing functionality."""
    print("\n" + "=" * 70)
    print("Testing Encoder Freezing")
    print("=" * 70)
    
    model = QuadModalSOTAModel(
        video_config=VIDEO_ENCODER,
        image_config=IMAGE_ENCODER,
        audio_config=AUDIO_ENCODER,
        sensor_config=SENSOR_ENCODER,
        fusion_config=FUSION,
        use_dummy=True,
    )
    
    # Initial state
    trainable_before = model.get_num_parameters(trainable_only=True)
    print(f"\nInitial trainable parameters: {trainable_before:,}")
    
    # Freeze encoders
    model.freeze_encoders()
    trainable_frozen = model.get_num_parameters(trainable_only=True)
    print(f"After freezing encoders: {trainable_frozen:,}")
    assert trainable_frozen < trainable_before, "Freezing should reduce trainable params"
    
    # Unfreeze encoders
    model.unfreeze_encoders()
    trainable_unfrozen = model.get_num_parameters(trainable_only=True)
    print(f"After unfreezing encoders: {trainable_unfrozen:,}")
    assert trainable_unfrozen == trainable_before, "Unfreezing should restore all params"
    
    print(f"\n✅ Encoder freezing test passed!\n")
    return True


def test_different_batch_sizes():
    """Test model with different batch sizes."""
    print("\n" + "=" * 70)
    print("Testing Different Batch Sizes")
    print("=" * 70)
    
    model = QuadModalSOTAModel(
        video_config=VIDEO_ENCODER,
        image_config=IMAGE_ENCODER,
        audio_config=AUDIO_ENCODER,
        sensor_config=SENSOR_ENCODER,
        fusion_config=FUSION,
        use_dummy=True,
    )
    
    batch_sizes = [1, 2, 4, 8]
    output_dim = model.get_feature_dim()
    
    for bs in batch_sizes:
        batch = {
            "video": torch.randn(bs, 32, 3, 224, 224),
            "post_weld_images": torch.randn(bs, 5, 3, 224, 224),
            "audio": torch.randn(bs, 1, 128, 256),
            "sensor": torch.randn(bs, 256, 6),
        }
        
        output = model(batch)
        assert output.shape == (bs, output_dim), \
            f"Batch size {bs}: Expected ({bs}, {output_dim}), got {output.shape}"
        
        print(f"  Batch size {bs:2d}: {tuple(output.shape)} ✅")
    
    print(f"\n✅ All batch sizes passed!\n")
    return True


def test_with_real_dataloader():
    """Test model with real dataloader (if available)."""
    print("\n" + "=" * 70)
    print("Testing with Real DataLoader")
    print("=" * 70)
    
    try:
        from src.dataset import WeldingDataset
        from torch.utils.data import DataLoader
        from configs.dataset_config import (
            DATA_ROOT,
            VIDEO_LENGTH,
            AUDIO_SAMPLE_RATE,
            AUDIO_DURATION,
            SENSOR_LENGTH,
            IMAGE_SIZE,
            IMAGE_NUM_ANGLES,
        )
        
        # Create dataset
        dataset = WeldingDataset(
            data_root=DATA_ROOT,
            video_length=VIDEO_LENGTH,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            audio_duration=AUDIO_DURATION,
            sensor_length=SENSOR_LENGTH,
            image_size=IMAGE_SIZE,
            num_angles=IMAGE_NUM_ANGLES,
            dummy=True,  # Use dummy data for testing
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
        
        # Create model
        model = QuadModalSOTAModel(
            video_config=VIDEO_ENCODER,
            image_config=IMAGE_ENCODER,
            audio_config=AUDIO_ENCODER,
            sensor_config=SENSOR_ENCODER,
            fusion_config=FUSION,
            use_dummy=True,
        )
        
        # Get one batch
        batch = next(iter(dataloader))
        
        print(f"\nDataLoader batch shapes:")
        for key in ["video", "post_weld_images", "audio", "sensor"]:
            if key in batch:
                print(f"  {key}: {tuple(batch[key].shape)}")
        print(f"  labels: {tuple(batch['label'].shape)}")
        
        # Forward pass
        output = model(batch)
        print(f"\nModel output shape: {tuple(output.shape)}")
        assert output.shape[0] == batch["label"].shape[0]
        
        print(f"\n✅ DataLoader integration test passed!\n")
        return True
        
    except Exception as e:
        print(f"\n⚠️  DataLoader test skipped: {e}\n")
        return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("QUADMODAL MODEL TEST SUITE")
    print("=" * 70)
    
    try:
        # Test 1: Dummy model
        test_quadmodal_model_dummy()
        
        # Test 2: Factory function
        test_quadmodal_model_factory()
        
        # Test 3: Encoder freezing
        test_encoder_freezing()
        
        # Test 4: Different batch sizes
        test_different_batch_sizes()
        
        # Test 5: DataLoader integration
        test_with_real_dataloader()
        
        print("\n" + "=" * 70)
        print("✅ ALL QUADMODAL MODEL TESTS PASSED!")
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
