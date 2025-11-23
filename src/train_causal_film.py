"""
Training script for Causal-FiLM model.

Implements unsupervised anomaly detection training:
- Only "normal" samples are used for training
- Reconstruction loss + CLIP text constraint
- Evaluation on test set with anomaly scores
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import create_causal_film_model
from src.dataset import WeldingDataset
from src.losses import CausalFILMLoss
from configs.dataset_config import (
    DATA_ROOT, MANIFEST_PATH, VIDEO_LENGTH, AUDIO_SAMPLE_RATE, AUDIO_DURATION,
    SENSOR_LENGTH, IMAGE_SIZE, IMAGE_NUM_ANGLES
)
from configs.model_config import *
from configs.train_config import *



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

class CausalFiLMTrainer:
    """Trainer for Causal-FiLM model with unsupervised anomaly detection."""
    
    def __init__(self, config: dict, use_dummy: bool = False):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dict
            use_dummy: Whether to use dummy data and models
        """
        self.config = config
        self.use_dummy = use_dummy
        self.device = torch.device(config.get("device", "cuda"))
        
        # Create output directories
        self.output_dir = Path(OUTPUT_DIR)
        self.checkpoint_dir = Path(CHECKPOINT_DIR)
        self.log_dir = Path(LOG_DIR)
        
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')  # Lower is better for reconstruction loss
        
        # Early stopping
        self.early_stopping_patience = config.get("early_stopping_patience", 0)
        self.epochs_without_improvement = 0
        
        # Setup model, data, optimizer
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        self._setup_loss()
        
        # Mixed precision training
        self.use_amp = config.get("mixed_precision", False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Logging
        self.train_log = []
        self.val_log = []
    
    def _setup_model(self):
        """Setup Causal-FiLM model."""
        print("=" * 70)
        print("INITIALIZING CAUSAL-FILM MODEL")
        print("=" * 70)
        
        model_config = {
            "VIDEO_ENCODER": VIDEO_ENCODER,
            "IMAGE_ENCODER": IMAGE_ENCODER,
            "AUDIO_ENCODER": AUDIO_ENCODER,
            "CAUSAL_FILM": CAUSAL_FILM_CONFIG,
        }
        
        self.model = create_causal_film_model(model_config, use_dummy=self.use_dummy)
        self.model = self.model.to(self.device)
        
        # Ensure backbones are frozen
        self.model.freeze_backbones()
        
        total_params = self.model.get_num_parameters(trainable_only=False)
        trainable_params = self.model.get_num_parameters(trainable_only=True)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Output dimension: {self.model.get_feature_dim()}")
        print(f"  Device: {self.device}")
        print()
    
    def _setup_data(self):
        """Setup data loaders (only normal samples for training)."""
        print("=" * 70)
        print("INITIALIZING DATA LOADERS")
        print("=" * 70)
        
        # Training dataset - ONLY NORMAL SAMPLES
        train_dataset = WeldingDataset(
            data_root=DATA_ROOT,
            manifest_path=MANIFEST_PATH,
            split='train',
            video_length=VIDEO_LENGTH,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            audio_duration=AUDIO_DURATION,
            sensor_length=SENSOR_LENGTH,
            image_size=IMAGE_SIZE,
            num_angles=IMAGE_NUM_ANGLES,
            dummy=self.use_dummy,
            augment=self.config.get("use_augmentations", False),
        )
        
        # Filter to keep only normal samples
        # Heuristic (robust to WeldingDataset variations):
        # 1) label == 0 -> normal
        # 2) sample id/path contains 'good' or 'normal' -> normal
        # Prefer deriving from dataset ids/paths; do NOT rely on label_to_name attribute.
        normal_indices = []
        for idx in range(len(train_dataset)):
            label_val = None
            try:
                label_val = int(train_dataset._labels[idx])
            except Exception:
                label_val = None

            # Try to infer label string from dataset ids/paths
            id_str = ""
            try:
                id_str = str(train_dataset._ids[idx])
            except Exception:
                id_str = ""

            # Try splitting path to get category folder (common manifest layout)
            label_str = ""
            if id_str:
                # e.g., '2_good_weld_2_02-09-23_Fe410/04-01-23-0024-00'
                label_str = id_str.split('/')[0]

            # Fallback: if dataset provides a mapping attribute, use it safely
            if hasattr(train_dataset, 'label_to_name'):
                try:
                    label_str = train_dataset.label_to_name.get(train_dataset._labels[idx], label_str)
                except Exception:
                    pass

            if (label_val is not None and label_val == 0) or (isinstance(label_str, str) and ("good" in label_str.lower() or "normal" in label_str.lower())):
                normal_indices.append(idx)
        
        if len(normal_indices) == 0:
            print("  Warning: No normal samples found! Using all training samples.")
            normal_indices = list(range(len(train_dataset)))
        
        # Create subset dataset
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, normal_indices)
        
        print(f"  Train samples (normal only): {len(train_dataset)}")
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=self.config.get("pin_memory", True),
        )
        
        # Validation dataset (includes normal + anomalies for evaluation)
        val_dataset = WeldingDataset(
            data_root=DATA_ROOT,
            manifest_path=MANIFEST_PATH,
            split='test',
            video_length=VIDEO_LENGTH,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            audio_duration=AUDIO_DURATION,
            sensor_length=SENSOR_LENGTH,
            image_size=IMAGE_SIZE,
            num_angles=IMAGE_NUM_ANGLES,
            dummy=self.use_dummy,
            augment=False,
        )
        
        print(f"  Val samples (normal + anomalies): {len(val_dataset)}")
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=self.config.get("pin_memory", True),
        )
        
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print()
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        print("=" * 70)
        print("INITIALIZING OPTIMIZER")
        print("=" * 70)
        
        lr = self.config["learning_rate"]
        weight_decay = self.config.get("weight_decay", 1e-4)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["num_epochs"],
            eta_min=self.config.get("min_lr", 1e-7),
        )
        
        print(f"  Optimizer: AdamW")
        print(f"  Learning rate: {lr}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Scheduler: CosineAnnealingLR")
        print()
    
    def _setup_loss(self):
        """Setup Causal-FiLM loss function."""
        print("=" * 70)
        print("INITIALIZING LOSS FUNCTION")
        print("=" * 70)
        
        lambda_text = self.config.get("lambda_text", 0.1)
        
        self.criterion = CausalFILMLoss(
            lambda_text=lambda_text,
            clip_model_name="ViT-B/32",
            text_prompt="a normal weld",
            device=self.device,
        )
        
        print(f"  Loss: Causal-FiLM Loss")
        print(f"  Lambda (text): {lambda_text}")
        print()
    
    def _move_batch_to_device(self, batch: dict):
        """Move batch to device."""
        for key in list(batch.keys()):
            val = batch[key]
            if isinstance(val, torch.Tensor):
                if val.is_floating_point() and val.dtype != torch.float32:
                    val = val.float()
                batch[key] = val.to(self.device)
            elif isinstance(val, (list, tuple)):
                try:
                    arr = np.asarray(val)
                    t = torch.from_numpy(arr)
                    if t.is_floating_point() and t.dtype != torch.float32:
                        t = t.float()
                    batch[key] = t.to(self.device)
                except Exception:
                    batch[key] = val
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {"total": [], "recon_cos": [], "recon_l1": [], "clip_text": []}
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            self._move_batch_to_device(batch)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    output = self.model(batch)
                    loss_dict = self.criterion(
                        Z_result=output["Z_result"],
                        Z_result_pred=output["Z_result_pred"],
                    )
                    loss = loss_dict["total"]
            else:
                output = self.model(batch)
                loss_dict = self.criterion(
                    Z_result=output.get("Z_result"),
                    Z_result_pred=output.get("Z_result_pred"),
                )
                loss = loss_dict["total"]
            
            # Backward pass
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                # Gradient clipping
                if self.config.get("gradient_clip", 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["gradient_clip"]
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.get("gradient_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["gradient_clip"]
                    )
                self.optimizer.step()
            
            # Log losses
            epoch_losses["total"].append(loss.item())
            epoch_losses["recon_cos"].append(loss_dict["recon_cos"])
            epoch_losses["recon_l1"].append(loss_dict["recon_l1"])
            epoch_losses["clip_text"].append(loss_dict["clip_text"])
            
            self.global_step += 1
            
            # Print progress
            if (batch_idx + 1) % self.config.get("log_interval", 10) == 0:
                avg_total = np.mean(epoch_losses["total"][-10:])
                avg_cos = np.mean(epoch_losses["recon_cos"][-10:])
                avg_l1 = np.mean(epoch_losses["recon_l1"][-10:])
                avg_clip = np.mean(epoch_losses["clip_text"][-10:])
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"    Batch [{batch_idx+1:3d}/{len(self.train_loader):3d}] "
                      f"Loss: {avg_total:.4f} (Cos: {avg_cos:.4f}, L1: {avg_l1:.4f}, C: {avg_clip:.4f}) "
                      f"LR: {lr:.2e}")
        
        epoch_time = time.time() - epoch_start_time
        
        # Compute epoch statistics
        epoch_stats = {
            "epoch": epoch,
            "loss": np.mean(epoch_losses["total"]),
            "recon_cos_loss": np.mean(epoch_losses["recon_cos"]),
            "recon_l1_loss": np.mean(epoch_losses["recon_l1"]),
            "clip_text_loss": np.mean(epoch_losses["clip_text"]),
            "lr": self.optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time,
        }
        
        return epoch_stats
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate model."""
        self.model.eval()
        val_losses = {"total": [], "recon_cos": [], "recon_l1": [], "clip_text": []}
        anomaly_scores = []
        labels = []
        
        for batch in self.val_loader:
            self._move_batch_to_device(batch)
            
            # Forward pass
            output = self.model(batch)
            loss_dict = self.criterion(
                Z_result=output.get("Z_result"),
                Z_result_pred=output.get("Z_result_pred"),
            )
            
            val_losses["total"].append(loss_dict["total"].item())
            val_losses["recon_cos"].append(loss_dict["recon_cos"])
            val_losses["recon_l1"].append(loss_dict["recon_l1"])
            val_losses["clip_text"].append(loss_dict["clip_text"])
            
            # Compute anomaly scores
            scores = self.model.compute_anomaly_score(
                Z_result=output.get("Z_result"),
                Z_result_pred=output.get("Z_result_pred"),
            )
            anomaly_scores.append(scores.cpu())
            
            if "label" in batch:
                labels.append(batch["label"].cpu())
        
        # Compute validation statistics
        anomaly_scores_cat = torch.cat(anomaly_scores)
        val_stats = {
            "epoch": epoch,
            "loss": np.mean(val_losses["total"]),
            "recon_cos_loss": np.mean(val_losses["recon_cos"]),
            "recon_l1_loss": np.mean(val_losses["recon_l1"]),
            "clip_text_loss": np.mean(val_losses["clip_text"]),
            "mean_anomaly_score": anomaly_scores_cat.mean().item(),
            "std_anomaly_score": anomaly_scores_cat.std().item(),
            "min_anomaly_score": anomaly_scores_cat.min().item(),
            "max_anomaly_score": anomaly_scores_cat.max().item(),
        }
        
        return val_stats
    
    def train(self):
        """Main training loop."""
        print("=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Total epochs: {self.config['num_epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Lambda (CLIP): {self.config.get('lambda_text', 0.1)}")
        print(f"Early stopping patience: {self.early_stopping_patience}")
        print()
        
        start_time = time.time()
        
        for epoch in range(1, self.config["num_epochs"] + 1):
            epoch_header = f"Epoch {epoch}/{self.config['num_epochs']}"
            print("=" * 70)
            print(epoch_header)
            print("=" * 70)
            
            # Train
            train_stats = self.train_epoch(epoch)
            self.train_log.append(train_stats)
            
            print(f"\n  [TRAIN] Loss: {train_stats['loss']:.4f} | "
                  f"Cos: {train_stats['recon_cos_loss']:.4f} | "
                  f"L1: {train_stats['recon_l1_loss']:.4f} | "
                  f"CLIP: {train_stats['clip_text_loss']:.4f} | "
                  f"LR: {train_stats['lr']:.2e} | "
                  f"Time: {train_stats['epoch_time']:.1f}s")
            
            # Validate
            if epoch % self.config.get("val_interval", 1) == 0:
                val_stats = self.validate(epoch)
                self.val_log.append(val_stats)
                
                print(f"  [VAL]   Loss: {val_stats['loss']:.4f} | "
                      f"Cos: {val_stats['recon_cos_loss']:.4f} | "
                      f"L1: {val_stats['recon_l1_loss']:.4f} | "
                      f"CLIP: {val_stats['clip_text_loss']:.4f}")
                print(f"  [SCORE] Mean: {val_stats['mean_anomaly_score']:.4f} | "
                      f"Std: {val_stats['std_anomaly_score']:.4f} | "
                      f"Range: [{val_stats['min_anomaly_score']:.4f}, {val_stats['max_anomaly_score']:.4f}]")
                
                # Check for improvement
                current_metric = val_stats["loss"]
                if current_metric < self.best_metric:
                    improvement = self.best_metric - current_metric
                    self.best_metric = current_metric
                    self.epochs_without_improvement = 0
                    self._save_checkpoint(epoch, is_best=True)
                    print(f"  [BEST]  New best model saved! Loss improved by {improvement:.4f}")
                else:
                    self.epochs_without_improvement += 1
                    print(f"  [INFO]  No improvement for {self.epochs_without_improvement} epoch(s)")
                
                # Early stopping
                if self.early_stopping_patience > 0:
                    if self.epochs_without_improvement >= self.early_stopping_patience:
                        print(f"\n  [STOP]  Early stopping triggered after {epoch} epochs")
                        print(f"          No improvement for {self.early_stopping_patience} consecutive epochs")
                        break
            
            # Step scheduler
            self.scheduler.step()
            
            # Save checkpoint periodically
            if epoch % self.config.get("save_interval", 5) == 0:
                self._save_checkpoint(epoch, is_best=False)
                print(f"  [SAVE]  Checkpoint saved at epoch {epoch}")
            
            print()
        
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        print("=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total time: {hours:02d}h {minutes:02d}m {seconds:02d}s")
        print(f"Best validation loss: {self.best_metric:.4f}")
        print(f"Total epochs trained: {len(self.train_log)}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"Logs saved to: {self.log_dir}")
        print("=" * 70)
        
        # Save final logs
        self._save_logs()
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "train_log": self.train_log,
            "val_log": self.val_log,
        }
        
        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, path)
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, path)
    
    def _save_logs(self):
        """Save training logs."""
        log_file = self.log_dir / "training_log.json"
        with open(log_file, "w") as f:
            json.dump({
                "train_log": self.train_log,
                "val_log": self.val_log,
            }, f, indent=2, cls=NumpyEncoder)



def main():
    """Main training function."""
    import argparse
    parser = argparse.ArgumentParser(description="Train Causal-FiLM model")
    parser.add_argument("--dummy", action="store_true", help="Use dummy data and models")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    
    # Use config from train_config.py
    config = TRAIN_CONFIG.copy()
    
    # Override with command line args
    if args.config:
        cfg_path = Path(args.config)
        # If a python config file is provided (e.g., configs/train_config.py), import it
        if cfg_path.suffix == ".py":
            import importlib.util
            spec = importlib.util.spec_from_file_location("train_config_module", str(cfg_path))
            cfg_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cfg_mod)
            # Expect TRAIN_CONFIG dict in the python module
            if hasattr(cfg_mod, "TRAIN_CONFIG"):
                config.update(getattr(cfg_mod, "TRAIN_CONFIG"))
            else:
                raise RuntimeError(f"Python config {cfg_path} does not define TRAIN_CONFIG dict")
        else:
            import json
            with open(cfg_path) as f:
                config.update(json.load(f))
    
    # Create trainer
    trainer = CausalFiLMTrainer(config, use_dummy=args.dummy)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
