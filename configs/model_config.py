"""Model configuration."""

# Model paths
VIDEO_MODEL_PATH = "models/vjepa2-vitl-fpc64-256"
IMAGE_MODEL_PATH = "/root/work/models/dinov3-vith16plus-pretrain-lvd1689m"
AUDIO_MODEL_PATH = "models/ast-finetuned-audioset-14-14-0.443"

# Video encoder config
VIDEO_ENCODER = {
    "model_name": VIDEO_MODEL_PATH,
    "embed_dim": 1024,
    "freeze_backbone": True,  # Freeze pretrained weights for fine-tuning
}

# Image encoder config (DINOv3-vith16plus)
IMAGE_ENCODER = {
    "model_name": IMAGE_MODEL_PATH,
    "embed_dim": 1280,  # DINOv3-vit-h/16+ outputs 1280-dim features
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
    "embed_dim": 128,
    "num_heads": 8,
    "num_layers": 2,
    "dim_feedforward": 512,
    "dropout": 0.2,  # Moderate regularization (reduced from 0.4)
}

# Fusion config (Cross-Attention Fusion Module)
FUSION = {
    "video_dim": 1024,
    "image_dim": 768,
    "audio_dim": 768,
    "sensor_dim": 256,
    "hidden_dim": 384,
    "num_fusion_tokens": 2,
    "num_heads": 4,
    "dropout": 0.2,  # Moderate regularization (reduced from 0.4)
    "l2_normalize": True,  # L2 normalization for contrastive learning (recommended)
}

# Output config
NUM_CLASSES = 12  # Updated from 6 to match actual dataset (12 categories)

# Causal-FiLM config
CAUSAL_FILM_CONFIG = {
    "d_model": 512,  # Increased from 256 to compensate for dropout regularization
    "sensor_input_dim": 6,  # Sensor input channels
    "sensor_hidden_dim": 128,  # Increased from 64
    "decoder_num_layers": 2,  # Number of decoder layers
    "decoder_dropout": 0.2,  # Noisy bottleneck dropout
}
