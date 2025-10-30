"""Model configuration."""

# Model paths
VIDEO_MODEL_PATH = "models/vjepa2-vitl-fpc64-256"
IMAGE_MODEL_PATH = "models/dinov2-base"
AUDIO_MODEL_PATH = "models/ast-finetuned-audioset-14-14-0.443"

# Video encoder config
VIDEO_ENCODER = {
    "model_name": VIDEO_MODEL_PATH,
    "embed_dim": 1024,
    "freeze_backbone": True,  # Freeze pretrained weights for fine-tuning
}

# Image encoder config (DINOv2)
IMAGE_ENCODER = {
    "model_name": IMAGE_MODEL_PATH,
    "embed_dim": 768,
    "num_angles": 5,
    "aggregation": "mean",
    "freeze_backbone": True,  # Freeze pretrained weights for fine-tuning
}

# Audio encoder config
AUDIO_ENCODER = {
    "model_name": AUDIO_MODEL_PATH,
    "embed_dim": 768,
    "freeze_backbone": True,  # Freeze pretrained weights for fine-tuning
}

# Sensor encoder config (Transformer)
SENSOR_ENCODER = {
    "input_dim": 6,
    "embed_dim": 256,
    "num_heads": 8,
    "num_layers": 4,
    "dim_feedforward": 1024,
    "dropout": 0.2,  # Moderate regularization (reduced from 0.4)
}

# Fusion config (Cross-Attention Fusion Module)
FUSION = {
    "video_dim": 1024,
    "image_dim": 768,
    "audio_dim": 768,
    "sensor_dim": 256,
    "hidden_dim": 512,
    "num_fusion_tokens": 4,
    "num_heads": 8,
    "dropout": 0.2,  # Moderate regularization (reduced from 0.4)
    "l2_normalize": True,  # L2 normalization for contrastive learning (recommended)
}

# Output config
NUM_CLASSES = 12  # Updated from 6 to match actual dataset (12 categories)
