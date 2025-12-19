"""Evaluation script for Late Fusion baseline.

Evaluates trained audio and video models, optimizes fusion weights on validation set,
and reports final performance on test set.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from baselines.Late_Fusion.models import AudioAutoEncoder, VideoAutoEncoder, LateFusionModel
from baselines.Late_Fusion.config import AUDIO_CONFIG, VIDEO_CONFIG, FUSION_CONFIG, EVAL_CONFIG
from baselines.Late_Fusion.utils import (
    aggregate_audio_scores_expected_value,
    aggregate_video_scores_max_over_2s_ma,
    compute_standardization_stats,
    standardize_scores,
    compute_auc_per_class,
)
from src.dataset import WeldingDataset


def f1_score_max(labels: np.ndarray, scores: np.ndarray) -> float:
    precs, recs, _ = precision_recall_curve(labels, scores)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    return float(np.max(f1s)) if f1s.size else 0.0


def extract_anomaly_scores(
    model,
    dataloader,
    modality: str,
    device: str = "cuda",
):
    """Extract frame-level anomaly scores from a model.
    
    Args:
        model: Trained auto-encoder model
        dataloader: Data loader
        modality: "audio" or "video"
        device: Device to run on
        
    Returns:
        Tuple of (scores, labels, class_labels)
    """
    model = model.to(device)
    model.eval()
    
    all_scores = []
    all_labels = []
    all_class_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if modality == "audio":
                data = batch["audio"].to(device)
                data = data.squeeze(1)  # (batch, n_bins, time)
                scores = model.get_reconstruction_error(data)
            else:  # video
                data = batch["video"].to(device)
                scores = model.get_reconstruction_error(data)
            
            # Move to CPU and convert to numpy
            scores = scores.cpu().numpy()  # (batch, time_steps)
            labels = batch["label"].numpy()
            
            all_scores.extend(scores)
            all_labels.extend(labels)
            
            # Get class labels if available
            if "class_label" in batch:
                all_class_labels.extend(batch["class_label"].numpy())
            else:
                all_class_labels.extend(labels)
    
    return all_scores, np.array(all_labels), np.array(all_class_labels)


def aggregate_scores(
    frame_scores: list,
    modality: str,
    fps: float = 30.0,
):
    """Aggregate frame-level scores to sample-level.
    
    Args:
        frame_scores: List of frame-level score arrays
        modality: "audio" or "video"
        fps: Frame rate (for video)
        
    Returns:
        Array of sample-level scores
    """
    sample_scores = []
    
    for scores in frame_scores:
        if modality == "audio":
            # Expected value for audio
            score = aggregate_audio_scores_expected_value(scores)
        else:
            # Max over 2s-MA for video
            score = aggregate_video_scores_max_over_2s_ma(scores, fps=fps)
        
        sample_scores.append(score)
    
    return np.array(sample_scores)


def evaluate_single_modality(
    model,
    train_loader,
    test_loader,
    modality: str,
    device: str = "cuda",
):
    """Evaluate single modality model.
    
    Args:
        model: Trained model
        train_loader: Training set loader (for standardization)
        val_loader: Validation set loader
        test_loader: Test set loader
        modality: "audio" or "video"
        device: Device to run on
        
    Returns:
        Dictionary with results
    """
    train_frame_scores, _, _ = extract_anomaly_scores(model, train_loader, modality, device)
    test_frame_scores, test_labels, test_class_labels = extract_anomaly_scores(model, test_loader, modality, device)

    train_scores = aggregate_scores(train_frame_scores, modality)
    test_scores = aggregate_scores(test_frame_scores, modality)

    mean, std = compute_standardization_stats(train_scores)
    test_scores_std = standardize_scores(test_scores, mean, std)

    test_labels_binary = (np.array(test_labels) != 0).astype(int)
    test_auc = roc_auc_score(test_labels_binary, test_scores_std)
    test_ap = average_precision_score(test_labels_binary, test_scores_std)
    test_f1_max = f1_score_max(test_labels_binary, test_scores_std)

    test_auc_per_class = compute_auc_per_class(
        test_scores_std, test_labels_binary, test_class_labels
    )

    return {
        "modality": modality,
        "train_mean": float(mean),
        "train_std": float(std),
        "test_auc": float(test_auc),
        "test_ap": float(test_ap),
        "test_f1_max": float(test_f1_max),
        "test_scores": test_scores_std.tolist(),
        "test_labels": test_labels.tolist(),
        "test_auc_per_class": {k: (float(v) if v is not None else None) for k, v in test_auc_per_class.items()},
    }


def evaluate_fusion(
    audio_model,
    video_model,
    train_loader,
    test_loader,
    device: str = "cuda",
):
    """Evaluate multi-modal fusion.
    
    Args:
        audio_model: Trained audio model
        video_model: Trained video model
        train_loader: Training set loader
        test_loader: Test set loader
        device: Device to run on
        
    Returns:
        Dictionary with results
    """
    train_audio_frame, _, _ = extract_anomaly_scores(audio_model, train_loader, "audio", device)
    test_audio_frame, test_labels, test_class_labels = extract_anomaly_scores(audio_model, test_loader, "audio", device)

    train_video_frame, _, _ = extract_anomaly_scores(video_model, train_loader, "video", device)
    test_video_frame, _, _ = extract_anomaly_scores(video_model, test_loader, "video", device)

    train_audio = aggregate_scores(train_audio_frame, "audio")
    test_audio = aggregate_scores(test_audio_frame, "audio")
    train_video = aggregate_scores(train_video_frame, "video")
    test_video = aggregate_scores(test_video_frame, "video")

    audio_mean, audio_std = compute_standardization_stats(train_audio)
    video_mean, video_std = compute_standardization_stats(train_video)

    test_audio_std = standardize_scores(test_audio, audio_mean, audio_std)
    test_video_std = standardize_scores(test_video, video_mean, video_std)

    w_audio = float(FUSION_CONFIG.get("audio_weight", 0.5))
    w_video = float(FUSION_CONFIG.get("video_weight", 0.5))
    test_fused = w_audio * test_audio_std + w_video * test_video_std

    test_labels_binary = (np.array(test_labels) != 0).astype(int)

    test_auc_fused = roc_auc_score(test_labels_binary, test_fused)
    test_ap_fused = average_precision_score(test_labels_binary, test_fused)
    test_f1_max_fused = f1_score_max(test_labels_binary, test_fused)

    test_auc_per_class = compute_auc_per_class(
        test_fused, test_labels_binary, test_class_labels
    )

    return {
        "modality": "fusion",
        "audio_weight": w_audio,
        "video_weight": w_video,
        "audio_mean": float(audio_mean),
        "audio_std": float(audio_std),
        "video_mean": float(video_mean),
        "video_std": float(video_std),
        "test_auc": float(test_auc_fused),
        "test_ap": float(test_ap_fused),
        "test_f1_max": float(test_f1_max_fused),
        "test_scores": test_fused.tolist(),
        "test_labels": test_labels.tolist(),
        "test_auc_per_class": {k: (float(v) if v is not None else None) for k, v in test_auc_per_class.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Late Fusion baseline")
    parser.add_argument("--audio_checkpoint", type=str,
                        default="baselines/Late_Fusion/checkpoints/audio_autoencoder_best.pth",
                        help="Path to audio model checkpoint")
    parser.add_argument("--video_checkpoint", type=str,
                        default="baselines/Late_Fusion/checkpoints/video_autoencoder_best.pth",
                        help="Path to video model checkpoint")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to dataset root")
    parser.add_argument("--manifest", type=str, default="configs/manifest.csv",
                        help="Path to manifest file")
    parser.add_argument("--output_dir", type=str, default="baselines/Late_Fusion/results",
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--dummy", action="store_true",
                        help="Use dummy data for testing")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    
    train_dataset = WeldingDataset(
        root_dir=args.data_root,
        mode="dummy" if args.dummy else "real",
        split="train",
        manifest_path=args.manifest,
        audio_type="stft",
        audio_sr=AUDIO_CONFIG["sample_rate"],
        n_fft=AUDIO_CONFIG["n_fft"],
        hop_length=AUDIO_CONFIG["hop_length"],
        audio_frames=1024,
    )
    
    test_dataset = WeldingDataset(
        root_dir=args.data_root,
        mode="dummy" if args.dummy else "real",
        split="test",
        manifest_path=args.manifest,
        audio_type="stft",
        audio_sr=AUDIO_CONFIG["sample_rate"],
        n_fft=AUDIO_CONFIG["n_fft"],
        hop_length=AUDIO_CONFIG["hop_length"],
        audio_frames=1024,
    )
    
    # Audio STFT is large (8193 bins), reduce eval batch size to avoid GPU/CPU stalls
    audio_batch_size = AUDIO_CONFIG.get("batch_size_eval", max(1, AUDIO_CONFIG["batch_size"] // 4))
    train_loader = DataLoader(train_dataset, batch_size=audio_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=audio_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load models
    print("\nLoading models...")
    
    # Audio model - Load from checkpoint to get correct config
    if os.path.exists(args.audio_checkpoint):
        checkpoint = torch.load(args.audio_checkpoint, map_location=args.device)
        
        # Use config from checkpoint if available, otherwise use default
        if "config" in checkpoint:
            audio_config = checkpoint["config"]
            n_bins = audio_config["n_fft"] // 2 + 1
            bottleneck_dim = audio_config["bottleneck_dim"]
            hidden_channels = audio_config["hidden_channels"]
            num_conv_layers = audio_config["num_conv_layers"]
        else:
            # Fallback to default config
            n_bins = AUDIO_CONFIG["n_fft"] // 2 + 1
            bottleneck_dim = AUDIO_CONFIG["bottleneck_dim"]
            hidden_channels = AUDIO_CONFIG["hidden_channels"]
            num_conv_layers = AUDIO_CONFIG["num_conv_layers"]
        
        audio_model = AudioAutoEncoder(
            n_bins=n_bins,
            bottleneck_dim=bottleneck_dim,
            hidden_channels=hidden_channels,
            num_conv_layers=num_conv_layers,
        )
        
        audio_model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded audio model from {args.audio_checkpoint}")
        print(f"    n_bins={n_bins}, bottleneck_dim={bottleneck_dim}, "
              f"hidden_channels={hidden_channels}, num_conv_layers={num_conv_layers}")
    else:
        print(f"  Warning: Audio checkpoint not found at {args.audio_checkpoint}")
        # Create model with default config
        n_bins = AUDIO_CONFIG["n_fft"] // 2 + 1
        audio_model = AudioAutoEncoder(
            n_bins=n_bins,
            bottleneck_dim=AUDIO_CONFIG["bottleneck_dim"],
            hidden_channels=AUDIO_CONFIG["hidden_channels"],
            num_conv_layers=AUDIO_CONFIG["num_conv_layers"],
        )
    
    # Video model - Load from checkpoint to get correct config
    if os.path.exists(args.video_checkpoint):
        checkpoint = torch.load(args.video_checkpoint, map_location=args.device)
        
        # Use config from checkpoint if available, otherwise use default
        if "config" in checkpoint:
            video_config = checkpoint["config"]
            feature_dim = video_config["feature_dim"]
            encoder_layers = video_config["encoder_layers"]
            decoder_layers = video_config["decoder_layers"]
            dropout = video_config["dropout"]
        else:
            # Fallback to default config
            feature_dim = VIDEO_CONFIG["feature_dim"]
            encoder_layers = VIDEO_CONFIG["encoder_layers"]
            decoder_layers = VIDEO_CONFIG["decoder_layers"]
            dropout = VIDEO_CONFIG["dropout"]
        
        video_model = VideoAutoEncoder(
            feature_dim=feature_dim,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            dropout=dropout,
        )
        
        video_model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded video model from {args.video_checkpoint}")
        print(f"    feature_dim={feature_dim}, encoder_layers={encoder_layers}, "
              f"decoder_layers={decoder_layers}, dropout={dropout}")
    else:
        print(f"  Warning: Video checkpoint not found at {args.video_checkpoint}")
        # Create model with default config
        video_model = VideoAutoEncoder(
            feature_dim=VIDEO_CONFIG["feature_dim"],
            encoder_layers=VIDEO_CONFIG["encoder_layers"],
            decoder_layers=VIDEO_CONFIG["decoder_layers"],
            dropout=VIDEO_CONFIG["dropout"],
        )
    
    # Evaluate models
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    fusion_results = evaluate_fusion(
        audio_model, video_model, train_loader, test_loader, args.device
    )

    print(
        f"Test Split - I-AUROC: {fusion_results['test_auc']:.4f}, "
        f"AP: {fusion_results['test_ap']:.4f}, F1-max: {fusion_results['test_f1_max']:.4f}"
    )
    print("\nPer-Defect-Type I-AUROC:")
    print("=" * 60)
    for cls, auc in fusion_results["test_auc_per_class"].items():
        if auc is None:
            print(f"  {cls:40s}: N/A")
        else:
            print(f"  {cls:40s}: {auc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
