"""Configuration for Video Autoencoder and Late Fusion."""

import os
from .dataset_config import DATA_ROOT, MANIFEST_PATH, VIDEO_LENGTH, IMAGE_SIZE

# Video Autoencoder Training Config
VIDEO_AE_CONFIG = {
    "model_name": "/root/work/models/dinov2-base",
    "input_dim": 768,
    "hidden_dims": [128, 64], # Encoder structure: 768 -> 128 -> 64
    "batch_size": 8,
    "num_epochs": 30,
    "learning_rate": 1e-3,
    "device": "cuda",
    "save_path": "/root/autodl-tmp/outputs/checkpoints/best_video_ae.pth",
    "num_workers": 4,
    "use_wandb": True,
    "wandb_project": "weld-anomaly-detection",
    "wandb_entity": None,
    "wandb_run_name": "video-ae-training",
}

# Fusion Evaluation Config
FUSION_CONFIG = {
    "checkpoint_a": "/root/autodl-tmp/outputs/checkpoints/best_model.pth", # Causal-FiLM
    "checkpoint_b": "/root/autodl-tmp/outputs/checkpoints/best_video_ae.pth", # Video AE
    "batch_size": 8,
    "device": "cuda",
    "num_workers": 4,
}
