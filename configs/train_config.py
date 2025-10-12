"""Training configuration."""

# Training hyperparameters
TRAIN_CONFIG = {
    # Optimization
    "batch_size": 16,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "optimizer": "adamw",  # "adam", "adamw", "sgd"
    
    # Learning rate schedule
    "lr_scheduler": "cosine",  # "cosine", "step", "plateau", "none"
    "warmup_epochs": 5,
    "min_lr": 1e-6,
    
    # Loss
    "loss_type": "supcon",  # "supcon", "combined"
    "temperature": 0.07,
    "use_ce": False,
    "ce_weight": 0.0,
    "supcon_weight": 1.0,
    
    # Training strategy
    "freeze_encoders_epochs": 0,  # Freeze encoders for first N epochs (0 = no freeze)
    "gradient_clip": 1.0,  # Gradient clipping norm (0 = no clip)
    "early_stopping_patience": 10,  # Stop if val loss doesn't improve for N epochs (0 = no early stopping)
    
    # Data
    "num_workers": 4,
    "pin_memory": True,
    "prefetch_factor": 2,
    
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
