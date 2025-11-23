"""Plot confusion matrix using nearest-centroid predictions.

Usage examples (cmd.exe):
  python scripts/plot_confusion_matrix.py --checkpoint outputs/checkpoints/best_model.pth
  python scripts/plot_confusion_matrix.py --use-dummy --output outputs/plots/confusion.png

The script loads the Trainer (model + dataloaders), optionally loads a checkpoint,
extracts features for train and val, computes class centroids on train features,
predicts val labels by cosine similarity, and plots + saves a confusion matrix.
"""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
import json
import numpy as np

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

import torch

from src.train import Trainer
from configs.train_config import TRAIN_CONFIG


def load_checkpoint(model, ckpt_path: str, device: torch.device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)


def compute_centroids(features: np.ndarray, labels: np.ndarray):
    unique = np.unique(labels)
    centroids = {}
    for u in unique:
        mask = labels == u
        if mask.sum() == 0:
            continue
        centroids[int(u)] = features[mask].mean(axis=0)
    return centroids


def predict_nearest_centroid(centroids: dict, feats: np.ndarray, metric: str = 'cosine'):
    # centroids: label -> vector
    labels = sorted(centroids.keys())
    C = np.stack([centroids[l] for l in labels], axis=0)  # (K, D)

    if metric == 'cosine':
        def norm(x):
            n = np.linalg.norm(x, axis=1, keepdims=True)
            n[n == 0] = 1.0
            return x / n
        feats_n = norm(feats)
        cent_n = norm(C)
        sims = feats_n.dot(cent_n.T)
        idx = sims.argmax(axis=1)
        preds = np.array([labels[i] for i in idx])
    else:
        # Euclidean
        dists = np.sqrt(((feats[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))
        idx = dists.argmin(axis=1)
        preds = np.array([labels[i] for i in idx])

    return preds


def plot_and_save(cm: np.ndarray, labels: list, out_path: str):
    plt.figure(figsize=(8, 6))
    if _HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    else:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot confusion matrix using nearest-centroid')
    # Prefer server path if present; user provided server location.
    parser.add_argument('--checkpoint', type=str, default='/root/autodl-tmp/outputs/checkpoints/best_model.pth', help='Path to model checkpoint (optional)')
    parser.add_argument('--use-dummy', action='store_true', help='Use dummy data and encoders')
    parser.add_argument('--device', type=str, default=None, help='Device to use, e.g. cuda or cpu')
    parser.add_argument('--output', type=str, default='outputs/confusion_matrix.png', help='Output image path')
    parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'euclidean'], help='Distance metric')
    args = parser.parse_args()

    cfg = TRAIN_CONFIG.copy()
    if args.device:
        cfg['device'] = args.device
    else:
        cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # small override: batch size may affect memory; use config batch_size
    trainer = Trainer(cfg, use_dummy=args.use_dummy)

    device = torch.device(cfg.get('device', 'cpu'))

    # Optionally load checkpoint; if missing, fall back to dummy mode so the script
    # still produces a usable confusion matrix for smoke-testing.
    ckpt_path = args.checkpoint
    if ckpt_path and os.path.isfile(ckpt_path):
        try:
            load_checkpoint(trainer.model, ckpt_path, device)
            print(f"Loaded checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    else:
        if ckpt_path:
            print(f"Checkpoint not found at '{ckpt_path}'. Falling back to dummy data (use --checkpoint to specify).")
        else:
            print("No checkpoint specified. Using dummy data.")
        # Recreate trainer in dummy mode
        trainer = Trainer(cfg, use_dummy=True)

    # Extract train and val features
    print("Extracting train features...")
    train_feats, train_labels = trainer.extract_features(trainer.train_loader)
    print("Extracting val features...")
    val_feats, val_labels = trainer.extract_features(trainer.val_loader)

    if train_feats.size == 0 or val_feats.size == 0:
        print("No features extracted (empty dataset). Exiting.")
        return

    # Compute centroids on train
    centroids = compute_centroids(train_feats, train_labels)
    # Predict on val
    preds = predict_nearest_centroid(centroids, val_feats, metric=args.metric)

    # Confusion matrix
    labels_sorted = sorted(list(centroids.keys()))
    cm = confusion_matrix(val_labels, preds, labels=labels_sorted)

    # Save image
    plot_and_save(cm, labels_sorted, args.output)
    print(f"Confusion matrix saved to: {args.output}")

    # Print classification report
    print("\nClassification report (on val set):")
    print(classification_report(val_labels, preds, labels=labels_sorted, zero_division=0))


if __name__ == '__main__':
    main()
