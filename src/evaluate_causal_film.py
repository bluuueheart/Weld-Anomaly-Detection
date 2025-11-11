"""
Evaluation script for Causal-FiLM model.

Computes IMAGE-LEVEL anomaly detection metrics:
- I-AUROC / AUC: Area Under ROC Curve
- I-AP / I-mAP: Average Precision (AP)
- Accuracy (Acc): Overall classification accuracy
- F1-Score (F1): Harmonic mean of precision and recall
- FDR: False Discovery Rate (FP / (FP + TP))
- MDR: Missed Detection Rate (FN / (FN + TP))
- Precision, Recall at optimal threshold

NOTE: Pixel-level metrics (P-AUROC, P-AUPRO) have been removed due to lack of
pixel-level ground truth annotations. Only reliable image-level metrics are computed.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import create_causal_film_model
from src.dataset import WeldingDataset
from configs.dataset_config import (
    DATA_ROOT, MANIFEST_PATH, VIDEO_LENGTH, AUDIO_SAMPLE_RATE, AUDIO_DURATION,
    SENSOR_LENGTH, IMAGE_SIZE, IMAGE_NUM_ANGLES
)
from configs.model_config import *
from configs.train_config import OUTPUT_DIR


class CausalFiLMEvaluator:
    """Evaluator for Causal-FiLM using anomaly scores."""
    
    def __init__(
        self,
        checkpoint_path: str,
        use_dummy: bool = False,
        device: str = "cuda",
    ):
        """
        Initialize evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            use_dummy: Whether to use dummy data
            device: Device to run on
        """
        self.checkpoint_path = checkpoint_path
        self.use_dummy = use_dummy
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load model from checkpoint."""
        print("=" * 70)
        print("LOADING CAUSAL-FILM MODEL")
        print("=" * 70)
        
        # Create model
        model_config = {
            "VIDEO_ENCODER": VIDEO_ENCODER,
            "IMAGE_ENCODER": IMAGE_ENCODER,
            "AUDIO_ENCODER": AUDIO_ENCODER,
            "CAUSAL_FILM": CAUSAL_FILM_CONFIG,
        }
        
        self.model = create_causal_film_model(model_config, use_dummy=self.use_dummy)
        
        # Load checkpoint with weights_only=False to support full checkpoint format
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best Metric: {checkpoint.get('best_metric', 'N/A')}")
        print(f"  Device: {self.device}")
        print()
    
    @torch.no_grad()
    def extract_anomaly_scores(self, dataloader: DataLoader):
        """
        Extract image-level anomaly scores.
        
        Args:
            dataloader: DataLoader for evaluation
            
        Returns:
            scores: Image-level anomaly scores (N,)
            labels: Binary labels (N,) - 0=normal, 1=anomaly
            raw_labels: Original class labels (N,)
        """
        print("Extracting image-level anomaly scores...")
        
        all_scores = []
        all_labels = []
        all_raw_labels = []
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass
            output = self.model(batch, return_encodings=True)
            
            # Image-level anomaly scores
            scores = self.model.compute_anomaly_score(
                output["Z_result"],
                output["Z_result_pred"],
            )
            
            all_scores.append(scores.cpu())
            
            if "label" in batch:
                labels = batch["label"].cpu()
                all_raw_labels.append(labels)
                
                # Convert to binary: 0=normal, 1=anomaly
                binary_labels = (labels != 0).long()
                all_labels.append(binary_labels)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        scores = torch.cat(all_scores).numpy()
        labels = torch.cat(all_labels).numpy() if all_labels else None
        raw_labels = torch.cat(all_raw_labels).numpy() if all_raw_labels else None
        
        print(f"  Total samples: {len(scores)}")
        if labels is not None:
            print(f"  Normal samples: {(labels == 0).sum()}")
            print(f"  Anomaly samples: {(labels == 1).sum()}")
        print()
        
        return scores, labels, raw_labels
    
    def compute_metrics(self, scores, labels):
        """
        Compute image-level anomaly detection metrics.
        
        Args:
            scores: Anomaly scores (N,)
            labels: Binary labels (N,) - 0=normal, 1=anomaly
            
        Returns:
            metrics: Dictionary of metrics
        """
        if labels is None:
            print("Warning: No labels available for metric computation")
            return {}
        
        print("Computing image-level metrics...")
        
        from collections import OrderedDict
        metrics = OrderedDict()
        
        # 1. I-AUROC / AUC (Area Under ROC Curve)
        auroc = roc_auc_score(labels, scores)
        metrics["I-AUROC"] = auroc
        metrics["AUC"] = auroc  # Same as I-AUROC
        print(f"  I-AUROC / AUC: {auroc:.4f}")
        
        # 2. I-AP / I-mAP (Average Precision)
        ap = average_precision_score(labels, scores)
        metrics["I-AP"] = ap
        metrics["I-mAP"] = ap  # Same as I-AP (single class)
        print(f"  I-AP / I-mAP: {ap:.4f}")
        
        # 3. Find optimal threshold using Youden's J statistic
        fpr, tpr, thresholds = roc_curve(labels, scores)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        metrics["optimal_threshold"] = float(optimal_threshold)
        
        # 4. Compute predictions at optimal threshold
        predictions = (scores >= optimal_threshold).astype(int)
        
        # 5. Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        metrics["TP"] = int(tp)
        metrics["FP"] = int(fp)
        metrics["TN"] = int(tn)
        metrics["FN"] = int(fn)
        
        # 6. Accuracy (Acc)
        accuracy = accuracy_score(labels, predictions)
        metrics["Accuracy"] = accuracy
        metrics["Acc"] = accuracy
        print(f"  Accuracy (Acc): {accuracy:.4f}")
        
        # 7. F1-Score
        f1 = f1_score(labels, predictions)
        metrics["F1-Score"] = f1
        metrics["F1"] = f1
        print(f"  F1-Score (F1): {f1:.4f}")
        
        # 8. Precision and Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics["Precision"] = precision
        metrics["Recall"] = recall
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        # 9. FDR (False Discovery Rate) = FP / (FP + TP)
        fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
        metrics["FDR"] = fdr
        print(f"  FDR (False Discovery Rate): {fdr:.4f}")
        
        # 10. MDR (Missed Detection Rate) = FN / (FN + TP)
        mdr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        metrics["MDR"] = mdr
        print(f"  MDR (Missed Detection Rate): {mdr:.4f}")
        
        print(f"\n  Optimal Threshold: {optimal_threshold:.4f}")
        print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        
        print()
        return metrics
    
    def evaluate(self, split: str = "test"):
        """
        Evaluate model on dataset split.
        
        Args:
            split: Dataset split to evaluate ('test' or 'train')
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        print("=" * 70)
        print(f"EVALUATING ON {split.upper()} SPLIT")
        print("=" * 70)
        print()
        
        # Create dataset
        dataset = WeldingDataset(
            data_root=DATA_ROOT,
            manifest_path=MANIFEST_PATH,
            split=split,
            video_length=VIDEO_LENGTH,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            audio_duration=AUDIO_DURATION,
            sensor_length=SENSOR_LENGTH,
            image_size=IMAGE_SIZE,
            num_angles=IMAGE_NUM_ANGLES,
            dummy=self.use_dummy,
            augment=False,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        # Extract image-level anomaly scores
        scores, labels, raw_labels = self.extract_anomaly_scores(dataloader)
        
        # Compute image-level metrics
        metrics = self.compute_metrics(scores, labels)
        
        # Per-class statistics
        if raw_labels is not None:
            print("Per-class anomaly scores:")
            unique_labels = np.unique(raw_labels)
            for label in unique_labels:
                mask = raw_labels == label
                class_scores = scores[mask]
                # Safely get class name from dataset id/path
                class_name = f"Class_{label}"
                try:
                    if hasattr(dataset, 'label_to_name'):
                        class_name = dataset.label_to_name.get(label, class_name)
                    elif hasattr(dataset, '_ids') and len(dataset._ids) > 0:
                        # Try to infer from first sample with this label
                        for idx, lbl in enumerate(dataset._labels if hasattr(dataset, '_labels') else []):
                            if lbl == label:
                                sample_id = dataset._ids[idx]
                                class_name = sample_id.split('/')[0] if '/' in sample_id else class_name
                                break
                except Exception:
                    pass
                print(f"  {class_name}: mean={class_scores.mean():.4f}, "
                      f"std={class_scores.std():.4f}, "
                      f"min={class_scores.min():.4f}, "
                      f"max={class_scores.max():.4f}")
            print()
        
        return metrics, scores, labels, raw_labels


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Causal-FiLM model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "test"],
                       help="Dataset split to evaluate")
    parser.add_argument("--dummy", action="store_true",
                       help="Use dummy data and models")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save results (default: outputs/eval_results.json)")
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = CausalFiLMEvaluator(
        checkpoint_path=args.checkpoint,
        use_dummy=args.dummy,
        device=args.device,
    )
    
    # Evaluate
    metrics, scores, labels, raw_labels = evaluator.evaluate(split=args.split)
    
    # Save results
    output_path = args.output or os.path.join(OUTPUT_DIR, "eval_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    metrics_serializable = convert_to_serializable(metrics)
    
    # Add metadata to results
    results = {
        "metrics": metrics_serializable,
        "metadata": {
            "checkpoint": args.checkpoint,
            "split": args.split,
            "note": "All metrics are image-level (no pixel-level annotations available)"
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("=" * 70)
    print(f"Results saved to: {output_path}")
    print("=" * 70)
    print()
    # Safe formatter: if value is numeric, format to 4 decimals; otherwise print raw
    def _fmt(key, default='N/A'):
        v = metrics_serializable.get(key, default)
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    print("KEY RESULTS (Image-Level Metrics):")
    print(f"  • I-AUROC / AUC:      {_fmt('I-AUROC')}" )
    print(f"  • I-AP / I-mAP:       {_fmt('I-AP')}" )
    print(f"  • Accuracy (Acc):     {_fmt('Accuracy')}" )
    print(f"  • F1-Score (F1):      {_fmt('F1-Score')}" )
    print(f"  • FDR:                {_fmt('FDR')}" )
    print(f"  • MDR:                {_fmt('MDR')}" )
    print(f"  • Precision:          {_fmt('Precision')}" )
    print(f"  • Recall:             {_fmt('Recall')}" )
    print("=" * 70)


if __name__ == "__main__":
    main()
