"""Model configuration."""

# Model paths
VIDEO_MODEL_PATH = "models/vjepa2-vitl-fpc64-256"
IMAGE_MODEL_PATH = "models/dinov2-base"
AUDIO_MODEL_PATH = "models/ast-finetuned-audioset-14-14-0.443"

# Video encoder config
VIDEO_ENCODER = {
    "model_name": "facebook/vjepa2-vitl-fpc64-256",
    "embed_dim": 1024,
    "freeze_backbone": False,
}

# Image encoder config (DINOv2)
IMAGE_ENCODER = {
    "model_name": "facebook/dinov2-base",
    "embed_dim": 768,
    "num_angles": 5,
    "aggregation": "mean",
    "freeze_backbone": False,
}

# Audio encoder config
AUDIO_ENCODER = {
    "model_name": "MIT/ast-finetuned-audioset-14-14-0.443",
    "embed_dim": 768,
    "freeze_backbone": False,
}

# Sensor encoder config (Transformer)
SENSOR_ENCODER = {
    "input_dim": 6,
    "embed_dim": 256,
    "num_heads": 8,
    "num_layers": 4,
    "dim_feedforward": 1024,
    "dropout": 0.1,
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
    "dropout": 0.1,
}

# Output config
NUM_CLASSES = 6
