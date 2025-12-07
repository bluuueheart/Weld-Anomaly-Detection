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
from sklearn.metrics import roc_auc_score

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from baselines.Late_Fusion.models import AudioAutoEncoder, VideoAutoEncoder, LateFusionModel
from baselines.Late_Fusion.config import AUDIO_CONFIG, VIDEO_CONFIG, FUSION_CONFIG, EVAL_CONFIG
from baselines.Late_Fusion.utils import (
    aggregate_audio_scores_expected_value,
    aggregate_video_scores_max_over_2s_ma,
    compute_standardization_stats,
    standardize_scores,
    optimize_fusion_weights,
    compute_auc_per_class,
    plot_roc_curves,
)
from src.dataset import WeldingDataset


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
    val_loader,
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
    print(f"\nEvaluating {modality.upper()} model...")
    
    # Extract scores
    train_frame_scores, train_labels, _ = extract_anomaly_scores(
        model, train_loader, modality, device
    )
    val_frame_scores, val_labels, val_class_labels = extract_anomaly_scores(
        model, val_loader, modality, device
    )
    test_frame_scores, test_labels, test_class_labels = extract_anomaly_scores(
        model, test_loader, modality, device
    )
    
    # Aggregate to sample-level
    train_scores = aggregate_scores(train_frame_scores, modality)
    val_scores = aggregate_scores(val_frame_scores, modality)
    test_scores = aggregate_scores(test_frame_scores, modality)
    
    # Compute standardization stats from training set
    mean, std = compute_standardization_stats(train_scores)
    print(f"  Training set stats: mean={mean:.6f}, std={std:.6f}")
    
    # Standardize scores
    val_scores_std = standardize_scores(val_scores, mean, std)
    test_scores_std = standardize_scores(test_scores, mean, std)
    
    # Compute AUC on validation set
    val_auc = roc_auc_score(val_labels, val_scores_std)
    print(f"  Validation AUC: {val_auc:.4f}")
    
    # Compute AUC on test set
    test_auc = roc_auc_score(test_labels, test_scores_std)
    print(f"  Test AUC: {test_auc:.4f}")
    
    # Per-class AUC on test set
    if EVAL_CONFIG["report_per_class"]:
        test_auc_per_class = compute_auc_per_class(
            test_scores_std, test_labels, test_class_labels
        )
        print("  Test AUC per class:")
        for cls, auc in test_auc_per_class.items():
            print(f"    {cls}: {auc:.4f}")
    
    results = {
        "modality": modality,
        "train_mean": float(mean),
        "train_std": float(std),
        "val_auc": float(val_auc),
        "test_auc": float(test_auc),
        "test_scores": test_scores_std.tolist(),
        "test_labels": test_labels.tolist(),
    }
    
    if EVAL_CONFIG["report_per_class"]:
        results["test_auc_per_class"] = {k: float(v) for k, v in test_auc_per_class.items()}
    
    return results


def evaluate_fusion(
    audio_model,
    video_model,
    train_loader,
    val_loader,
    test_loader,
    device: str = "cuda",
):
    """Evaluate multi-modal fusion.
    
    Args:
        audio_model: Trained audio model
        video_model: Trained video model
        train_loader: Training set loader
        val_loader: Validation set loader
        test_loader: Test set loader
        device: Device to run on
        
    Returns:
        Dictionary with results
    """
    print("\nEvaluating FUSION model...")
    
    # Extract and aggregate scores for both modalities
    # Audio
    train_audio_frame, train_labels, _ = extract_anomaly_scores(
        audio_model, train_loader, "audio", device
    )
    val_audio_frame, val_labels, val_class_labels = extract_anomaly_scores(
        audio_model, val_loader, "audio", device
    )
    test_audio_frame, test_labels, test_class_labels = extract_anomaly_scores(
        audio_model, test_loader, "audio", device
    )
    
    train_audio = aggregate_scores(train_audio_frame, "audio")
    val_audio = aggregate_scores(val_audio_frame, "audio")
    test_audio = aggregate_scores(test_audio_frame, "audio")
    
    # Video
    train_video_frame, _, _ = extract_anomaly_scores(
        video_model, train_loader, "video", device
    )
    val_video_frame, _, _ = extract_anomaly_scores(
        video_model, val_loader, "video", device
    )
    test_video_frame, _, _ = extract_anomaly_scores(
        video_model, test_loader, "video", device
    )
    
    train_video = aggregate_scores(train_video_frame, "video")
    val_video = aggregate_scores(val_video_frame, "video")
    test_video = aggregate_scores(test_video_frame, "video")
    
    # Standardize using training set statistics
    audio_mean, audio_std = compute_standardization_stats(train_audio)
    video_mean, video_std = compute_standardization_stats(train_video)
    
    print(f"  Audio stats: mean={audio_mean:.6f}, std={audio_std:.6f}")
    print(f"  Video stats: mean={video_mean:.6f}, std={video_std:.6f}")
    
    val_audio_std = standardize_scores(val_audio, audio_mean, audio_std)
    val_video_std = standardize_scores(val_video, video_mean, video_std)
    test_audio_std = standardize_scores(test_audio, audio_mean, audio_std)
    test_video_std = standardize_scores(test_video, video_mean, video_std)
    
    # Optimize fusion weights on validation set
    best_w, best_val_auc, (w_audio, w_video) = optimize_fusion_weights(
        val_audio_std,
        val_video_std,
        val_labels,
        step=FUSION_CONFIG["grid_search_step"],
    )
    
    print(f"  Optimal weights: audio={w_audio:.4f}, video={w_video:.4f}")
    print(f"  Validation AUC (fused): {best_val_auc:.4f}")
    
    # Apply optimal weights to test set
    test_fused = w_audio * test_audio_std + w_video * test_video_std
    test_auc_fused = roc_auc_score(test_labels, test_fused)
    
    print(f"  Test AUC (fused): {test_auc_fused:.4f}")
    
    # Per-class AUC on test set
    if EVAL_CONFIG["report_per_class"]:
        test_auc_per_class = compute_auc_per_class(
            test_fused, test_labels, test_class_labels
        )
        print("  Test AUC per class (fused):")
        for cls, auc in test_auc_per_class.items():
            print(f"    {cls}: {auc:.4f}")
    
    results = {
        "modality": "fusion",
        "audio_weight": float(w_audio),
        "video_weight": float(w_video),
        "audio_mean": float(audio_mean),
        "audio_std": float(audio_std),
        "video_mean": float(video_mean),
        "video_std": float(video_std),
        "val_auc": float(best_val_auc),
        "test_auc": float(test_auc_fused),
        "test_scores": test_fused.tolist(),
        "test_labels": test_labels.tolist(),
    }
    
    if EVAL_CONFIG["report_per_class"]:
        results["test_auc_per_class"] = {k: float(v) for k, v in test_auc_per_class.items()}
    
    return results


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
    )
    
    val_dataset = WeldingDataset(
        root_dir=args.data_root,
        mode="dummy" if args.dummy else "real",
        split="val",
        manifest_path=args.manifest,
    )
    
    test_dataset = WeldingDataset(
        root_dir=args.data_root,
        mode="dummy" if args.dummy else "real",
        split="test",
        manifest_path=args.manifest,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load models
    print("\nLoading models...")
    
    # Audio model
    n_bins = AUDIO_CONFIG["n_fft"] // 2 + 1
    audio_model = AudioAutoEncoder(
        n_bins=n_bins,
        bottleneck_dim=AUDIO_CONFIG["bottleneck_dim"],
        hidden_channels=AUDIO_CONFIG["hidden_channels"],
        num_conv_layers=AUDIO_CONFIG["num_conv_layers"],
    )
    
    if os.path.exists(args.audio_checkpoint):
        checkpoint = torch.load(args.audio_checkpoint, map_location=args.device)
        audio_model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded audio model from {args.audio_checkpoint}")
    else:
        print(f"  Warning: Audio checkpoint not found at {args.audio_checkpoint}")
    
    # Video model
    video_model = VideoAutoEncoder(
        feature_dim=VIDEO_CONFIG["feature_dim"],
        encoder_layers=VIDEO_CONFIG["encoder_layers"],
        decoder_layers=VIDEO_CONFIG["decoder_layers"],
        dropout=VIDEO_CONFIG["dropout"],
    )
    
    if os.path.exists(args.video_checkpoint):
        checkpoint = torch.load(args.video_checkpoint, map_location=args.device)
        video_model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded video model from {args.video_checkpoint}")
    else:
        print(f"  Warning: Video checkpoint not found at {args.video_checkpoint}")
    
    # Evaluate models
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    # Audio
    audio_results = evaluate_single_modality(
        audio_model, train_loader, val_loader, test_loader, "audio", args.device
    )
    
    # Video
    video_results = evaluate_single_modality(
        video_model, train_loader, val_loader, test_loader, "video", args.device
    )
    
    # Fusion
    fusion_results = evaluate_fusion(
        audio_model, video_model, train_loader, val_loader, test_loader, args.device
    )
    
    # Save results
    results = {
        "audio": audio_results,
        "video": video_results,
        "fusion": fusion_results,
        "timestamp": datetime.now().isoformat(),
    }
    
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Plot ROC curves
    if EVAL_CONFIG["plot_roc"]:
        scores_dict = {
            "Audio": np.array(audio_results["test_scores"]),
            "Video": np.array(video_results["test_scores"]),
            "Fusion": np.array(fusion_results["test_scores"]),
        }
        labels = np.array(fusion_results["test_labels"])
        
        roc_path = os.path.join(args.output_dir, "roc_curves.png")
        plot_roc_curves(scores_dict, labels, save_path=roc_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Audio Test AUC:  {audio_results['test_auc']:.4f}")
    print(f"Video Test AUC:  {video_results['test_auc']:.4f}")
    print(f"Fusion Test AUC: {fusion_results['test_auc']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
