"""Late Fusion baseline for welding anomaly detection.

This module implements the Late Fusion approach from the original paper,
combining separate audio and video auto-encoders for anomaly detection.
"""

from .models import AudioAutoEncoder, VideoAutoEncoder, LateFusionModel
from .config import AUDIO_CONFIG, VIDEO_CONFIG, FUSION_CONFIG, TRAIN_CONFIG, EVAL_CONFIG
from .utils import (
    compute_stft,
    aggregate_audio_scores_expected_value,
    aggregate_video_scores_max_over_2s_ma,
    compute_standardization_stats,
    standardize_scores,
    optimize_fusion_weights,
    compute_auc_per_class,
    plot_roc_curves,
    EarlyStopping,
)

__all__ = [
    # Models
    "AudioAutoEncoder",
    "VideoAutoEncoder",
    "LateFusionModel",
    # Config
    "AUDIO_CONFIG",
    "VIDEO_CONFIG",
    "FUSION_CONFIG",
    "TRAIN_CONFIG",
    "EVAL_CONFIG",
    # Utils
    "compute_stft",
    "aggregate_audio_scores_expected_value",
    "aggregate_video_scores_max_over_2s_ma",
    "compute_standardization_stats",
    "standardize_scores",
    "optimize_fusion_weights",
    "compute_auc_per_class",
    "plot_roc_curves",
    "EarlyStopping",
]

__version__ = "1.0.0"
__author__ = "GitHub Copilot"
__description__ = "Late Fusion baseline implementation for welding anomaly detection"
