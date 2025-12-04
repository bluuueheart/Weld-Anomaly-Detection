"""
Evaluation script for quad-modal welding anomaly detection using k-NN.

Implements feature extraction and k-NN classification for model evaluation.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import create_quadmodal_model
from src.dataset import WeldingDataset
from configs.dataset_config import (
    DATA_ROOT, MANIFEST_PATH, VIDEO_LENGTH, AUDIO_SAMPLE_RATE, AUDIO_DURATION,
    SENSOR_LENGTH, IMAGE_SIZE, IMAGE_NUM_ANGLES
)
from configs.model_config import *
from configs.train_config import OUTPUT_DIR


class Evaluator:
    """Evaluator using k-NN classification on extracted features."""
    
    def __init__(
        self,
        checkpoint_path: str,
        k: int = 5,
        metric: str = "cosine",
        use_dummy: bool = False,
        device: str = "cuda",
    ):
        """
        Initialize evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            k: Number of neighbors for k-NN
            metric: Distance metric ('cosine', 'euclidean', 'manhattan')
            use_dummy: Whether to use dummy data
            device: Device to run on
        """
        self.checkpoint_path = checkpoint_path
        self.k = k
        self.metric = metric
        self.use_dummy = use_dummy
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load model from checkpoint."""
        print("=" * 70)
        print("LOADING MODEL")
        print("=" * 70)
        
        # Create model
        model_config = {
            "VIDEO_ENCODER": VIDEO_ENCODER,
            "IMAGE_ENCODER": IMAGE_ENCODER,
            "AUDIO_ENCODER": AUDIO_ENCODER,
            "SENSOR_ENCODER": SENSOR_ENCODER,
            "FUSION": FUSION,
        }
        
        self.model = create_quadmodal_model(model_config, use_dummy=self.use_dummy)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Device: {self.device}")
        print()
        
    @torch.no_grad()
    def extract_features(self, dataloader: DataLoader):
        """
        Extract features from dataset.
        
        Args:
            dataloader: DataLoader for dataset
            
        Returns:
            features: (N, feature_dim) numpy array
            labels: (N,) numpy array
        """
        all_features = []
        all_labels = []
        
        for batch in dataloader:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Extract features
            features = self.model(batch)
            
            # Store
            all_features.append(features.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())
        
        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        return features, labels
    
    def evaluate(self, train_loader: DataLoader, test_loader: DataLoader):
        """
        Evaluate model using k-NN classification.
        
        Args:
            train_loader: Training set DataLoader
            test_loader: Test set DataLoader
            
        Returns:
            results: Dictionary of evaluation metrics
        """
        print("=" * 70)
        print("EXTRACTING FEATURES")
        print("=" * 70)
        
        # Extract training features
        print("Extracting training features...")
        train_features, train_labels = self.extract_features(train_loader)
        print(f"  Train features: {train_features.shape}")
        print(f"  Train labels: {train_labels.shape}")
        
        # Extract test features
        print("Extracting test features...")
        test_features, test_labels = self.extract_features(test_loader)
        print(f"  Test features: {test_features.shape}")
        print(f"  Test labels: {test_labels.shape}")
        print()
        
        # Train k-NN classifier
        print("=" * 70)
        print("TRAINING k-NN CLASSIFIER")
        print("=" * 70)
        print(f"  k = {self.k}")
        print(f"  metric = {self.metric}")
        
        knn = KNeighborsClassifier(
            n_neighbors=self.k,
            metric=self.metric,
            n_jobs=-1,
        )
        knn.fit(train_features, train_labels)
        print("  ✅ k-NN trained")
        print()
        
        # Predict on test set
        print("=" * 70)
        print("EVALUATING ON TEST SET")
        print("=" * 70)
        
        predictions = knn.predict(test_features)
        
        # Compute metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, average="weighted", zero_division=0)
        recall = recall_score(test_labels, predictions, average="weighted", zero_division=0)
        f1 = f1_score(test_labels, predictions, average="weighted", zero_division=0)
        
        print(f"  [TEST] Accuracy:  {accuracy:.4f}")
        print(f"  [TEST] Precision: {precision:.4f}")
        print(f"  [TEST] Recall:    {recall:.4f}")
        print(f"  [TEST] F1-Score:  {f1:.4f}")
        print()
        
        # Classification report
        print("=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)
        print(classification_report(test_labels, predictions, zero_division=0))
        
        # Confusion matrix
        print("=" * 70)
        print("CONFUSION MATRIX")
        print("=" * 70)
        cm = confusion_matrix(test_labels, predictions)
        print(cm)
        print()
        
        # Prepare results
        results = {
            "test/accuracy": float(accuracy),
            "test/precision": float(precision),
            "test/recall": float(recall),
            "test/f1_score": float(f1),
            "test/confusion_matrix": cm.tolist(),
            "k": self.k,
            "metric": self.metric,
            "train_samples": len(train_labels),
            "test_samples": len(test_labels),
        }
        
        return results


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate quad-modal model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/root/autodl-tmp/outputs/checkpoints/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of neighbors for k-NN",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean", "manhattan"],
        help="Distance metric for k-NN",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy data and models",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON (default: outputs/eval_results.json)",
    )
    
    args = parser.parse_args()
    
    # Setup output path
    if args.output is None:
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / "eval_results.json")
    
    # Create evaluator
    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        k=args.k,
        metric=args.metric,
        use_dummy=args.dummy,
    )
    
    # Create data loaders
    print("=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)
    
    train_dataset = WeldingDataset(
        data_root=DATA_ROOT,
        manifest_path=MANIFEST_PATH,
        split='train',  # Use train split from manifest.csv
        video_length=VIDEO_LENGTH,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        audio_duration=AUDIO_DURATION,
        sensor_length=SENSOR_LENGTH,
        image_size=IMAGE_SIZE,
        num_angles=IMAGE_NUM_ANGLES,
        dummy=args.dummy,
    )
    
    test_dataset = WeldingDataset(
        data_root=DATA_ROOT,
        manifest_path=MANIFEST_PATH,
        split='test',  # Use test split from manifest.csv
        video_length=VIDEO_LENGTH,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        audio_duration=AUDIO_DURATION,
        sensor_length=SENSOR_LENGTH,
        image_size=IMAGE_SIZE,
        num_angles=IMAGE_NUM_ANGLES,
        dummy=args.dummy,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=test_dataset.collate_fn,
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print()
    
    # Evaluate
    results = evaluator.evaluate(train_loader, test_loader)
    
    # Add metadata
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results["checkpoint"] = args.checkpoint
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Results saved to: {args.output}")
    print()


if __name__ == "__main__":
    main()
