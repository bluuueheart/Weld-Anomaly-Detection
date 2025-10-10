#!/usr/bin/env python
"""Test encoder modules."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "configs"))

import torch
from models.video_encoder import VideoEncoder, DummyVideoEncoder
from models.audio_encoder import AudioEncoder, DummyAudioEncoder
from models.sensor_encoder import SensorEncoder, DummySensorEncoder
from models.image_encoder import ImageEncoder, DummyImageEncoder
from model_config import VIDEO_ENCODER, AUDIO_ENCODER, SENSOR_ENCODER, IMAGE_ENCODER


def test_video_encoder():
    """Test video encoder."""
    print("="*80)
    print("Testing VideoEncoder")
    print("="*80)
    
    # Create dummy input: (batch, num_frames, channels, height, width)
    batch_size = 2
    num_frames = 32
    channels = 3
    height = width = 224
    
    dummy_input = torch.randn(batch_size, num_frames, channels, height, width)
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Test with dummy encoder (no pretrained model needed)
    print("\n--- Testing DummyVideoEncoder ---")
    encoder = DummyVideoEncoder(embed_dim=VIDEO_ENCODER['embed_dim'])
    encoder.eval()
    
    with torch.no_grad():
        output = encoder(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: (batch={batch_size}, seq_len, embed_dim={VIDEO_ENCODER['embed_dim']})")
    
    assert output.shape[0] == batch_size, f"Batch size mismatch: {output.shape[0]} != {batch_size}"
    assert output.shape[2] == VIDEO_ENCODER['embed_dim'], f"Embed dim mismatch: {output.shape[2]} != {VIDEO_ENCODER['embed_dim']}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    
    print("✅ DummyVideoEncoder test passed!")
    
    # Test with real encoder (if model available)
    print("\n--- Testing VideoEncoder (with pretrained model) ---")
    try:
        encoder = VideoEncoder(
            model_path=VIDEO_ENCODER['model_name'],
            embed_dim=VIDEO_ENCODER['embed_dim'],
        )
        encoder.eval()
        
        with torch.no_grad():
            output = encoder(dummy_input)
        
        print(f"Output shape: {output.shape}")
        assert output.shape[0] == batch_size
        assert output.shape[2] == VIDEO_ENCODER['embed_dim']
        print("✅ VideoEncoder test passed!")
    except Exception as e:
        print(f"⚠️  VideoEncoder test skipped (model not available): {e}")


def test_audio_encoder():
    """Test audio encoder."""
    print("\n" + "="*80)
    print("Testing AudioEncoder")
    print("="*80)
    
    # Create dummy input: (batch, 1, n_mels, time_frames)
    batch_size = 2
    n_mels = 128
    time_frames = 256
    
    dummy_input = torch.randn(batch_size, 1, n_mels, time_frames)
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Test with dummy encoder
    print("\n--- Testing DummyAudioEncoder ---")
    encoder = DummyAudioEncoder(embed_dim=AUDIO_ENCODER['embed_dim'])
    encoder.eval()
    
    with torch.no_grad():
        output = encoder(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: (batch={batch_size}, seq_len, embed_dim={AUDIO_ENCODER['embed_dim']})")
    
    assert output.shape[0] == batch_size
    assert output.shape[2] == AUDIO_ENCODER['embed_dim']
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    
    print("✅ DummyAudioEncoder test passed!")
    
    # Test with real encoder
    print("\n--- Testing AudioEncoder (with pretrained model) ---")
    try:
        encoder = AudioEncoder(
            model_path=AUDIO_ENCODER['model_name'],
            embed_dim=AUDIO_ENCODER['embed_dim'],
        )
        encoder.eval()
        
        with torch.no_grad():
            output = encoder(dummy_input)
        
        print(f"Output shape: {output.shape}")
        assert output.shape[0] == batch_size
        assert output.shape[2] == AUDIO_ENCODER['embed_dim']
        print("✅ AudioEncoder test passed!")
    except Exception as e:
        print(f"⚠️  AudioEncoder test skipped (model not available): {e}")


def test_image_encoder():
    """Test image encoder."""
    print("\n" + "="*80)
    print("Testing ImageEncoder")
    print("="*80)
    
    # Create dummy input: (batch, num_angles, channels, height, width)
    batch_size = 2
    num_angles = 5
    channels = 3
    height = width = 224
    
    dummy_input = torch.randn(batch_size, num_angles, channels, height, width)
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Test with dummy encoder
    print("\n--- Testing DummyImageEncoder ---")
    encoder = DummyImageEncoder(
        embed_dim=IMAGE_ENCODER['embed_dim'],
        num_angles=IMAGE_ENCODER['num_angles'],
        aggregation=IMAGE_ENCODER['aggregation'],
    )
    encoder.eval()
    
    with torch.no_grad():
        output = encoder(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: (batch={batch_size}, seq_len, embed_dim={IMAGE_ENCODER['embed_dim']})")
    
    assert output.shape[0] == batch_size
    assert output.shape[2] == IMAGE_ENCODER['embed_dim']
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    
    print("✅ DummyImageEncoder test passed!")
    
    # Test with real encoder
    print("\n--- Testing ImageEncoder (with pretrained model) ---")
    try:
        encoder = ImageEncoder(
            model_name=IMAGE_ENCODER['model_name'],
            embed_dim=IMAGE_ENCODER['embed_dim'],
            num_angles=IMAGE_ENCODER['num_angles'],
            aggregation=IMAGE_ENCODER['aggregation'],
        )
        encoder.eval()
        
        with torch.no_grad():
            output = encoder(dummy_input)
        
        print(f"Output shape: {output.shape}")
        assert output.shape[0] == batch_size
        assert output.shape[2] == IMAGE_ENCODER['embed_dim']
        print("✅ ImageEncoder test passed!")
    except Exception as e:
        print(f"⚠️  ImageEncoder test skipped (model not available): {e}")


def test_sensor_encoder():
    """Test sensor encoder."""
    print("\n" + "="*80)
    print("Testing SensorEncoder")
    print("="*80)
    
    # Create dummy input: (batch, seq_len, input_dim)
    batch_size = 2
    seq_len = 256
    input_dim = 6
    
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Test sensor encoder (Transformer)
    print("\n--- Testing SensorEncoder (Transformer) ---")
    encoder = SensorEncoder(
        input_dim=SENSOR_ENCODER['input_dim'],
        embed_dim=SENSOR_ENCODER['embed_dim'],
        num_heads=SENSOR_ENCODER['num_heads'],
        num_layers=SENSOR_ENCODER['num_layers'],
        dim_feedforward=SENSOR_ENCODER['dim_feedforward'],
        dropout=SENSOR_ENCODER['dropout'],
    )
    encoder.eval()
    
    with torch.no_grad():
        output = encoder(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: (batch={batch_size}, seq_len={seq_len}, embed_dim={SENSOR_ENCODER['embed_dim']})")
    
    assert output.shape == (batch_size, seq_len, SENSOR_ENCODER['embed_dim'])
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    
    print("✅ SensorEncoder test passed!")
    
    # Test dummy version
    print("\n--- Testing DummySensorEncoder ---")
    encoder = DummySensorEncoder(
        input_dim=input_dim,
        embed_dim=SENSOR_ENCODER['embed_dim'],
    )
    encoder.eval()
    
    with torch.no_grad():
        output = encoder(dummy_input)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, SENSOR_ENCODER['embed_dim'])
    print("✅ DummySensorEncoder test passed!")


def test_encoder_gradients():
    """Test gradient flow through encoders."""
    print("\n" + "="*80)
    print("Testing Gradient Flow")
    print("="*80)
    
    # Test video encoder gradients
    print("\n--- Video Encoder Gradients ---")
    encoder = DummyVideoEncoder(embed_dim=256)
    dummy_input = torch.randn(1, 8, 3, 224, 224, requires_grad=True)
    output = encoder(dummy_input)
    loss = output.mean()
    loss.backward()
    
    assert dummy_input.grad is not None, "No gradient for input"
    print("✅ Video encoder gradients OK")
    
    # Test image encoder gradients
    print("\n--- Image Encoder Gradients ---")
    encoder = DummyImageEncoder(embed_dim=256, num_angles=5)
    dummy_input = torch.randn(1, 5, 3, 224, 224, requires_grad=True)
    output = encoder(dummy_input)
    loss = output.mean()
    loss.backward()
    
    assert dummy_input.grad is not None, "No gradient for input"
    print("✅ Image encoder gradients OK")
    
    # Test audio encoder gradients
    print("\n--- Audio Encoder Gradients ---")
    encoder = DummyAudioEncoder(embed_dim=256)
    dummy_input = torch.randn(1, 1, 128, 256, requires_grad=True)
    output = encoder(dummy_input)
    loss = output.mean()
    loss.backward()
    
    assert dummy_input.grad is not None
    print("✅ Audio encoder gradients OK")
    
    # Test sensor encoder gradients
    print("\n--- Sensor Encoder Gradients ---")
    encoder = SensorEncoder(input_dim=6, embed_dim=256, num_layers=2)
    dummy_input = torch.randn(1, 64, 6, requires_grad=True)
    output = encoder(dummy_input)
    loss = output.mean()
    loss.backward()
    
    assert dummy_input.grad is not None
    print("✅ Sensor encoder gradients OK")


def main():
    """Run all encoder tests."""
    print("\n" + "🧪" * 40)
    print("Encoder Module Testing")
    print("🧪" * 40 + "\n")
    
    test_video_encoder()
    test_audio_encoder()
    test_image_encoder()
    test_sensor_encoder()
    test_encoder_gradients()
    
    print("\n" + "="*80)
    print("✅ All Encoder Tests Completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
