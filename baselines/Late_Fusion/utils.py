"""Utilities for Late Fusion baseline.

Includes:
- Data preprocessing (STFT for audio)
- Score aggregation methods (Expected Value, Max over 2s-MA)
- Evaluation metrics (AUC calculation)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from scipy import signal
from sklearn.metrics import roc_auc_score, roc_curve


def compute_stft(
    audio: np.ndarray,
    sample_rate: int = 192000,
    n_fft: int = 16384,
    hop_length: int = 8192,
) -> np.ndarray:
    """Compute Short-Time Fourier Transform for audio.
    
    Args:
        audio: Audio waveform (samples,) or (batch, samples)
        sample_rate: Audio sampling rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        
    Returns:
        Magnitude spectrogram (n_bins, time_steps) or (batch, n_bins, time_steps)
    """
    if audio.ndim == 1:
        # Single audio
        f, t, Zxx = signal.stft(
            audio,
            fs=sample_rate,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
        )
        magnitude = np.abs(Zxx)
        return magnitude
    else:
        # Batch of audio
        spectrograms = []
        for i in range(audio.shape[0]):
            f, t, Zxx = signal.stft(
                audio[i],
                fs=sample_rate,
                nperseg=n_fft,
                noverlap=n_fft - hop_length,
            )
            spectrograms.append(np.abs(Zxx))
        return np.stack(spectrograms, axis=0)


def aggregate_audio_scores_expected_value(
    frame_scores: np.ndarray,
) -> float:
    """Aggregate frame-level audio scores using expected value.
    
    Args:
        frame_scores: Frame-level anomaly scores (time_steps,)
        
    Returns:
        Sample-level anomaly score (scalar)
    """
    return np.mean(frame_scores)


def aggregate_video_scores_max_over_2s_ma(
    frame_scores: np.ndarray,
    fps: float = 30.0,
    window_seconds: float = 2.0,
) -> float:
    """Aggregate frame-level video scores using Max over 2s moving average.
    
    Args:
        frame_scores: Frame-level anomaly scores (num_frames,)
        fps: Video frame rate
        window_seconds: Moving average window size in seconds
        
    Returns:
        Sample-level anomaly score (scalar)
    """
    window_size = int(fps * window_seconds)
    
    if len(frame_scores) < window_size:
        # If video is shorter than window, just return mean
        return np.mean(frame_scores)
    
    # Compute moving average
    ma_scores = np.convolve(
        frame_scores,
        np.ones(window_size) / window_size,
        mode='valid'
    )
    
    # Return maximum of moving average
    return np.max(ma_scores)


def compute_standardization_stats(
    scores: np.ndarray,
) -> Tuple[float, float]:
    """Compute mean and std for score standardization.
    
    Args:
        scores: Array of scores from training set
        
    Returns:
        Tuple of (mean, std)
    """
    mean = np.mean(scores)
    std = np.std(scores)
    return mean, std


def standardize_scores(
    scores: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """Standardize scores using pre-computed statistics.
    
    Args:
        scores: Scores to standardize
        mean: Mean from training set
        std: Std from training set
        
    Returns:
        Standardized scores
    """
    return (scores - mean) / (std + 1e-8)


def optimize_fusion_weights(
    audio_scores: np.ndarray,
    video_scores: np.ndarray,
    labels: np.ndarray,
    step: float = 0.01,
) -> Tuple[float, float, float]:
    """Optimize fusion weights using grid search on validation set.
    
    Args:
        audio_scores: Standardized audio scores (num_samples,)
        video_scores: Standardized video scores (num_samples,)
        labels: Binary labels (0=good, 1=defective)
        step: Grid search step size
        
    Returns:
        Tuple of (best_weight, best_auc, weights)
        - best_weight: Optimal weight for audio (w)
        - best_auc: AUC achieved with best weight
        - weights: (audio_weight, video_weight)
    """
    best_auc = 0.0
    best_w = 0.0
    
    weights = np.arange(0.0, 1.0 + step, step)
    
    for w in weights:
        fused_scores = w * audio_scores + (1 - w) * video_scores
        
        # Compute AUC
        try:
            auc = roc_auc_score(labels, fused_scores)
            if auc > best_auc:
                best_auc = auc
                best_w = w
        except ValueError:
            # Skip if all labels are the same
            continue
    
    return best_w, best_auc, (best_w, 1 - best_w)


def compute_auc_per_class(
    scores: np.ndarray,
    labels: np.ndarray,
    class_labels: np.ndarray,
) -> dict:
    """Compute AUC for each defect class separately.
    
    Args:
        scores: Anomaly scores (num_samples,)
        labels: Binary labels (0=good, 1=defective)
        class_labels: Specific defect class for each sample
        
    Returns:
        Dictionary mapping class name to AUC
    """
    results = {}
    
    # Overall AUC
    try:
        overall_auc = roc_auc_score(labels, scores)
        results['overall'] = overall_auc
    except ValueError:
        results['overall'] = 0.0
    
    # Per-class AUC (each defect vs. good)
    unique_classes = np.unique(class_labels)
    good_mask = (labels == 0)
    
    for cls in unique_classes:
        if cls == 0:  # Skip "good" class
            continue
        
        # Create binary labels: good (0) vs. this defect (1)
        cls_mask = (class_labels == cls)
        binary_mask = good_mask | cls_mask
        
        if binary_mask.sum() < 2:
            continue
        
        cls_scores = scores[binary_mask]
        cls_labels = labels[binary_mask]
        
        try:
            auc = roc_auc_score(cls_labels, cls_scores)
            results[f'class_{cls}'] = auc
        except ValueError:
            results[f'class_{cls}'] = 0.0
    
    return results


def plot_roc_curves(
    scores_dict: dict,
    labels: np.ndarray,
    save_path: Optional[str] = None,
):
    """Plot ROC curves for different models.
    
    Args:
        scores_dict: Dictionary mapping model name to scores
        labels: Binary labels
        save_path: Path to save plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping ROC plot")
        return
    
    plt.figure(figsize=(10, 8))
    
    for name, scores in scores_dict.items():
        try:
            fpr, tpr, _ = roc_curve(labels, scores)
            auc = roc_auc_score(labels, scores)
            plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})', linewidth=2)
        except ValueError:
            continue
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Late Fusion Baseline', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 50, min_delta: float = 0.0):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Validation loss
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
