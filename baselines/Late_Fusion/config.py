"""Configuration for Late Fusion baseline model.

This configuration follows the paper's specifications for audio and video
auto-encoders used in the late fusion approach.
"""

# Audio Auto-Encoder Configuration
AUDIO_CONFIG = {
    # Input parameters
    "sample_rate": 192000,  # 192 kHz as specified in paper
    "n_fft": 16384,  # Best performing FFT window size from hyperparameter search
    "hop_length": 8192,  # 50% of FFT window
    "n_bins": None,  # Will be calculated from n_fft: n_fft // 2 + 1 = 8193
    
    # Model architecture (Table 5 in paper)
    "bottleneck_dim": 48,  # Best performing bottleneck dimension
    "hidden_channels": 1024,
    "num_conv_layers": 3,  # Number of 1024->1024 conv layers
    
    # Training parameters
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "optimizer": "adam",
    "loss": "mse",
    
    # Buffer parameters for online inference
    "latency_ms": 42.7,  # hop_length / sample_rate * 1000
    "buffer_size": 98304,  # hop_length * (10 + n_fft / hop_length)
}

# Video Auto-Encoder Configuration
VIDEO_CONFIG = {
    # Stage 1: Feature extraction with SlowFast
    "slowfast_model": "slowfast_r101_4x16x1_256e_kinetics400_rgb",
    "slowfast_checkpoint": "slowfast_r101_4x16x1_256e_kinetics400_rgb_20210218-d8b58813.pth",
    "freeze_slowfast": True,  # Freeze SlowFast weights during training
    "window_size": 64,  # 64 frames ~ 2 seconds at 30 FPS
    "feature_dim": 2304,  # SlowFast output feature dimension
    
    # Stage 2: Auto-encoder for anomaly detection (Table 6 in paper)
    "encoder_layers": [2304, 512, 256, 128, 64, 64],  # Last is bottleneck
    "decoder_layers": [64, 64, 128, 256, 512, 2304],
    "dropout": 0.5,  # Dropout probability
    
    # Training parameters
    "batch_size": 32,
    "num_epochs": 1000,  # Maximum epochs with early stopping
    "learning_rate": 5e-4,
    "optimizer": "adam",
    "loss": "mse",
    "early_stopping_patience": 50,
    
    # Inference parameters
    "latency_ms": 1000,  # ~1 second due to 64-frame window
}

# Multi-modal Fusion Configuration
FUSION_CONFIG = {
    # Aggregation methods
    "audio_aggregation": "expected_value",  # Expected value for audio
    "video_aggregation": "max_over_2s_ma",  # Max over 2-second moving average for video
    
    # Fusion parameters
    "fusion_method": "weighted_sum",
    "audio_weight": 0.37,  # w in paper (optimized on validation set)
    "video_weight": 0.63,  # 1-w in paper
    "grid_search_step": 0.01,  # Step size for weight optimization
    
    # Standardization (using training set statistics)
    "standardize": True,
}

# Training Strategy
TRAIN_CONFIG = {
    # Data split (must match paper's specification)
    "train_samples": {"good": 576},
    "val_samples": {"good": 122, "defective": 1610},
    "test_samples": {"good": 121, "defective": 1611},
    
    # Random seed for reproducibility
    "seed": 42,
    
    # Common settings
    "num_workers": 4,
    "pin_memory": True,
}

# Evaluation Configuration
EVAL_CONFIG = {
    # Metrics
    "primary_metric": "auc",  # Area Under ROC Curve
    "report_per_class": True,  # Report AUC for each defect type
    
    # Visualization
    "plot_roc": True,
    "save_predictions": True,
}
