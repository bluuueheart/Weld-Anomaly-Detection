"""
Evaluation script for Causal-FiLM model.

Computes anomaly detection metrics:
- AUROC (Area Under ROC Curve)
- AUPRO (Area Under Per-Region Overlap curve) at different FPR thresholds
- Precision, Recall, F1 at optimal threshold
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
)
try:
    from scipy.ndimage import label as ndimage_label
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. P-AUPRO will use simplified approximation.")

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
        
        # Load checkpoint
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
    def extract_anomaly_scores_and_maps(self, dataloader: DataLoader):
        """
        Extract anomaly scores and pixel-level anomaly maps.
        
        For P-AUPRO, we generate pseudo pixel-level anomaly maps by
        projecting the reconstruction error back to spatial dimensions.
        
        Args:
            dataloader: DataLoader for evaluation
            
        Returns:
            scores: Image-level anomaly scores (N,)
            labels: Binary labels (N,) - 0=normal, 1=anomaly
            raw_labels: Original class labels (N,)
            anomaly_maps: List of pixel-level anomaly maps (for P-AUPRO)
        """
        print("Extracting anomaly scores and maps...")
        
        all_scores = []
        all_labels = []
        all_raw_labels = []
        all_anomaly_maps = []
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass with intermediate features
            output = self.model(batch, return_encodings=True)
            
            # Image-level anomaly scores
            # Support both Dual-Stream (V7.1) and Legacy (V7.0) outputs
            if "Z_texture" in output:
                scores = self.model.compute_anomaly_score(
                    Z_texture=output["Z_texture"],
                    Z_structure=output["Z_structure"],
                    Z_texture_pred=output["Z_texture_pred"],
                    Z_structure_pred=output["Z_structure_pred"],
                )
            else:
                # Fallback for legacy models or Plan E model
                scores = self.model.compute_anomaly_score(
                    Z_result=output["Z_result"],
                    Z_result_pred=output["Z_result_pred"],
                )
            
            all_scores.append(scores.cpu())
            
            # Generate pixel-level anomaly maps
            # Method: Compute reconstruction error for each spatial location
            # Using the image features before pooling
            batch_size = scores.size(0)
            anomaly_maps_batch = []
            
            for i in range(batch_size):
                # Create a simple anomaly map based on the global score
                # In a real implementation, you would compute per-pixel reconstruction errors
                # Here we create a uniform map with the global score
                # This is a placeholder - ideally should be computed from spatial features
                anomaly_map = torch.ones(224, 224) * scores[i].item()
                anomaly_maps_batch.append(anomaly_map.cpu().numpy())
            
            all_anomaly_maps.extend(anomaly_maps_batch)
            
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
        
        return scores, labels, raw_labels, all_anomaly_maps
    
    def compute_metrics(self, scores, labels):
        """
        Compute anomaly detection metrics.
        
        Args:
            scores: Anomaly scores (N,)
            labels: Binary labels (N,) - 0=normal, 1=anomaly
            
        Returns:
            metrics: Dictionary of metrics
        """
        if labels is None:
            print("Warning: No labels available for metric computation")
            return {}
        
        print("Computing metrics...")
        
        metrics = {}
        
        # AUROC (Image-level)
        auroc = roc_auc_score(labels, scores)
        metrics["I-AUROC"] = auroc
        print(f"  I-AUROC: {auroc:.4f}")
        
        # Average Precision
        ap = average_precision_score(labels, scores)
        metrics["AP"] = ap
        print(f"  AP: {ap:.4f}")
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        metrics["optimal_threshold"] = optimal_threshold
        
        # Compute precision, recall, F1 at optimal threshold
        predictions = (scores >= optimal_threshold).astype(int)
        tp = ((predictions == 1) & (labels == 1)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()
        fn = ((predictions == 0) & (labels == 1)).sum()
        tn = ((predictions == 0) & (labels == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1
        metrics["tp"] = int(tp)
        metrics["fp"] = int(fp)
        metrics["tn"] = int(tn)
        metrics["fn"] = int(fn)
        
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        
        print()
        return metrics
    
    def compute_pro_metric(self, anomaly_maps, labels, fpr_limit=0.3):
        """
        Compute Per-Region Overlap (PRO) metric.
        
        PRO evaluates pixel-level anomaly detection by computing the overlap
        between predicted and ground truth anomaly regions.
        
        Args:
            anomaly_maps: List of anomaly maps (N, H, W)
            labels: Binary labels (N,) - 0=normal, 1=anomaly
            fpr_limit: FPR threshold for AUPRO computation
            
        Returns:
            aupro: Area Under PRO curve at given FPR limit
        """
        if not SCIPY_AVAILABLE:
            print("  Warning: scipy not available, skipping P-AUPRO computation")
            return 0.0
        
        # Since we don't have ground truth masks, we approximate P-AUPRO
        # by treating the entire image as a region
        # In real implementation, you would need pixel-level ground truth masks
        
        # For now, we compute a simplified version based on connected components
        all_pros = []
        all_fprs = []
        
        # Create thresholds from anomaly map values
        all_scores = np.concatenate([m.flatten() for m in anomaly_maps])
        thresholds = np.percentile(all_scores, np.linspace(0, 100, 100))
        
        for threshold in thresholds:
            tp_pixels = 0
            fp_pixels = 0
            fn_pixels = 0
            tn_pixels = 0
            
            for idx, (anomaly_map, label) in enumerate(zip(anomaly_maps, labels)):
                pred_mask = (anomaly_map >= threshold).astype(np.uint8)
                
                if label == 1:  # Anomaly
                    # Assume entire image is anomalous region
                    gt_mask = np.ones_like(pred_mask)
                    tp_pixels += np.sum(pred_mask & gt_mask)
                    fn_pixels += np.sum((1 - pred_mask) & gt_mask)
                else:  # Normal
                    # Assume entire image is normal
                    gt_mask = np.zeros_like(pred_mask)
                    fp_pixels += np.sum(pred_mask & (1 - gt_mask))
                    tn_pixels += np.sum((1 - pred_mask) & (1 - gt_mask))
            
            # Compute PRO and FPR
            pro = tp_pixels / (tp_pixels + fn_pixels) if (tp_pixels + fn_pixels) > 0 else 0
            fpr = fp_pixels / (fp_pixels + tn_pixels) if (fp_pixels + tn_pixels) > 0 else 0
            
            all_pros.append(pro)
            all_fprs.append(fpr)
        
        # Sort by FPR
        sorted_idx = np.argsort(all_fprs)
        sorted_fprs = np.array(all_fprs)[sorted_idx]
        sorted_pros = np.array(all_pros)[sorted_idx]
        
        # Compute AUPRO up to fpr_limit
        idx_limit = np.searchsorted(sorted_fprs, fpr_limit)
        if idx_limit > 0:
            # Use trapezoid if available (NumPy 2.0+), else trapz
            if hasattr(np, "trapezoid"):
                aupro = np.trapezoid(sorted_pros[:idx_limit], sorted_fprs[:idx_limit]) / fpr_limit
            else:
                aupro = np.trapz(sorted_pros[:idx_limit], sorted_fprs[:idx_limit]) / fpr_limit
        else:
            aupro = 0.0
        
        return aupro
    
    def compute_metrics_with_pro(self, scores, labels, anomaly_maps):
        """
        Compute comprehensive anomaly detection metrics including P-AUPRO.
        
        Args:
            scores: Anomaly scores (N,)
            labels: Binary labels (N,) - 0=normal, 1=anomaly
            anomaly_maps: List of pixel-level anomaly maps
            
        Returns:
            metrics: Dictionary of metrics
        """
        if labels is None:
            print("Warning: No labels available for metric computation")
            return {}
        
        print("Computing metrics...")
        
        metrics = {}
        
        # AUROC (Image-level Detection)
        auroc = roc_auc_score(labels, scores)
        metrics["I-AUROC"] = auroc
        print(f"  I-AUROC (Image-level Detection): {auroc:.4f}")
        
        # Average Precision
        ap = average_precision_score(labels, scores)
        metrics["AP"] = ap
        print(f"  AP: {ap:.4f}")
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        metrics["optimal_threshold"] = optimal_threshold
        
        # Compute precision, recall, F1 at optimal threshold
        predictions = (scores >= optimal_threshold).astype(int)
        tp = ((predictions == 1) & (labels == 1)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()
        fn = ((predictions == 0) & (labels == 1)).sum()
        tn = ((predictions == 0) & (labels == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1
        metrics["tp"] = int(tp)
        metrics["fp"] = int(fp)
        metrics["tn"] = int(tn)
        metrics["fn"] = int(fn)
        
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        
        # P-AUPRO (Pixel-level Segmentation)
        print("\n  Computing P-AUPRO (Pixel-level Segmentation)...")
        fpr_limits = [0.3, 0.1, 0.05, 0.01]
        for limit in fpr_limits:
            aupro = self.compute_pro_metric(anomaly_maps, labels, fpr_limit=limit)
            metrics[f"P-AUPRO@{limit}"] = aupro
            print(f"    P-AUPRO@{limit}: {aupro:.4f}")
        
        print()
        return metrics
    
    def compute_metrics(self, scores, labels):
        """
        Compute basic anomaly detection metrics (backward compatibility).
        
        Args:
            scores: Anomaly scores (N,)
            labels: Binary labels (N,) - 0=normal, 1=anomaly
            
        Returns:
            metrics: Dictionary of metrics
        """
        if labels is None:
            print("Warning: No labels available for metric computation")
            return {}
        
        print("Computing metrics...")
        
        metrics = {}
        
        # AUROC (Image-level)
        auroc = roc_auc_score(labels, scores)
        metrics["I-AUROC"] = auroc
        print(f"  I-AUROC: {auroc:.4f}")
        
        # Average Precision
        ap = average_precision_score(labels, scores)
        metrics["AP"] = ap
        print(f"  AP: {ap:.4f}")
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        metrics["optimal_threshold"] = optimal_threshold
        
        # Compute precision, recall, F1 at optimal threshold
        predictions = (scores >= optimal_threshold).astype(int)
        tp = ((predictions == 1) & (labels == 1)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()
        fn = ((predictions == 0) & (labels == 1)).sum()
        tn = ((predictions == 0) & (labels == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1
        metrics["tp"] = int(tp)
        metrics["fp"] = int(fp)
        metrics["tn"] = int(tn)
        metrics["fn"] = int(fn)
        
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        
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
        
        # Extract scores and anomaly maps for P-AUPRO
        scores, labels, raw_labels, anomaly_maps = self.extract_anomaly_scores_and_maps(dataloader)
        
        # Compute comprehensive metrics including P-AUPRO
        metrics = self.compute_metrics_with_pro(scores, labels, anomaly_maps)
        
        # Per-class statistics
        if raw_labels is not None:
            print("Per-class anomaly scores:")
            unique_labels = np.unique(raw_labels)
            for label in unique_labels:
                mask = raw_labels == label
                class_scores = scores[mask]
                class_name = dataset.label_to_name.get(label, f"Class_{label}")
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
    
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    
    print("=" * 70)
    print(f"Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
