"""Training configuration."""

# Training hyperparameters
TRAIN_CONFIG = {
    # Optimization
    "batch_size": 32,  # Increased for SupConLoss (need more positive pairs per batch)
    "num_epochs": 100,
    "learning_rate": 2e-5,  # Restored from 3e-5 for faster early convergence
    "weight_decay": 1e-2,  # Further increased to 0.01 (very strong L2)
    "optimizer": "adamw",  # "adam", "adamw", "sgd"
    
    # Learning rate schedule (with linear warmup)
    "lr_scheduler": "cosine_warmup",  # "cosine", "cosine_warmup", "step", "plateau", "none"
    "warmup_epochs": 5,  # Reduced from 10 (faster convergence to good solution)
    "warmup_start_lr": 1e-6,  # Higher start LR
    "min_lr": 1e-7,  # Lower floor to allow aggressive decay
    
    # Loss
    "loss_type": "supcon",  # "supcon", "combined"
    "temperature": 0.10,  # Increased from 0.07 (smoother similarity, less overfitting)
    "use_ce": False,
    "ce_weight": 0.0,
    "supcon_weight": 1.0,
    
    # Regularization via data augmentation
    "use_mixup": False,  # Feature-level mixup
    "mixup_alpha": 0.2,  # Beta distribution param (0.2 = conservative mixing)
    
    # Training strategy
    "freeze_encoders_epochs": 0,  # Freeze encoders for first N epochs (0 = no freeze)
    "gradient_clip": 0.5,  # Reduced from 1.0 (tighter gradient control)
    "early_stopping_patience": 8,  # Aggressive early stop (catch best at epoch 4-10)
    
    # Data
    "num_workers": 4,
    "pin_memory": True,
    "prefetch_factor": 2,
    # Data augmentation (applies to training split only)
    "use_augmentations": True,
    
    # Logging
    "log_interval": 10,  # Log every N batches
    "val_interval": 1,   # Validate every N epochs
    "save_interval": 5,  # Save checkpoint every N epochs
    
    # Checkpoint
    "resume": None,  # Path to checkpoint to resume from
    "save_best_only": True,
    "monitor_metric": "val_loss",  # Metric to monitor for best model
    
    # Device
    "device": "cuda",  # "cuda", "cpu", "cuda:0"
    "mixed_precision": False,  # Use AMP (Automatic Mixed Precision)
    
    # Reproducibility
    "seed": 42,
    "deterministic": False,
}

# Optimizer-specific configs
OPTIMIZER_CONFIGS = {
    "adam": {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    },
    "adamw": {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    },
    "sgd": {
        "momentum": 0.9,
        "nesterov": True,
    },
}

# Learning rate scheduler configs
SCHEDULER_CONFIGS = {
    "cosine": {
        "T_max": 100,  # Will be set to num_epochs
        "eta_min": 1e-6,
    },
    "step": {
        "step_size": 30,
        "gamma": 0.1,
    },
    "plateau": {
        "mode": "min",
        "factor": 0.1,
        "patience": 10,
        "threshold": 1e-4,
    },
}

# Output directories (user-requested base path)
# Saved outputs will be placed under /root/autodl-tmp/outputs
OUTPUT_DIR = "/root/autodl-tmp/outputs"
CHECKPOINT_DIR = "/root/autodl-tmp/outputs/checkpoints"
LOG_DIR = "/root/autodl-tmp/outputs/logs"
