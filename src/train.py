"""
Training script for quad-modal welding anomaly detection.

Implements supervised contrastive learning with clean logging.
"""

import os
import sys
import time
import json
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

from src.models import create_quadmodal_model
from src.dataset import WeldingDataset
from src.losses import SupConLoss, CombinedLoss
from src.samplers import StratifiedBatchSampler
from configs.dataset_config import *
from configs.model_config import *
from configs.train_config import *


class Trainer:
    """Trainer for quad-modal model with clean logging."""
    
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

        # Default debug flag (can be overridden later)
        self.debug = bool(config.get("debug_mode", False))

        # Create output directories
        self.output_dir = Path(OUTPUT_DIR)
        self.checkpoint_dir = Path(CHECKPOINT_DIR)
        self.log_dir = Path(LOG_DIR)
        
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
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
        """Setup model."""
        print("=" * 70)
        print("INITIALIZING MODEL")
        print("=" * 70)
        
        model_config = {
            "VIDEO_ENCODER": VIDEO_ENCODER,
            "IMAGE_ENCODER": IMAGE_ENCODER,
            "AUDIO_ENCODER": AUDIO_ENCODER,
            "SENSOR_ENCODER": SENSOR_ENCODER,
            "FUSION": FUSION,
        }
        
        self.model = create_quadmodal_model(model_config, use_dummy=self.use_dummy)
        self.model = self.model.to(self.device)
        
        total_params = self.model.get_num_parameters(trainable_only=False)
        trainable_params = self.model.get_num_parameters(trainable_only=True)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Output dimension: {self.model.get_feature_dim()}")
        print(f"  Device: {self.device}")
        print()
        
    def _setup_data(self):
        """Setup data loaders."""
        print("=" * 70)
        print("INITIALIZING DATA LOADERS")
        print("=" * 70)
        
        # Training dataset
        train_dataset = WeldingDataset(
            data_root=DATA_ROOT,
            video_length=VIDEO_LENGTH,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            audio_duration=AUDIO_DURATION,
            sensor_length=SENSOR_LENGTH,
            image_size=IMAGE_SIZE,
            num_angles=IMAGE_NUM_ANGLES,
            dummy=self.use_dummy,
        )
        
        if self.debug:
            train_label_counts = Counter(train_dataset._labels)
            print(f"  Train label distribution: {dict(sorted(train_label_counts.items()))}")

        drop_last = self.config.get("drop_last", self.config["batch_size"] > 1)

        # Use stratified sampler for better class balance in batches
        use_stratified = self.config.get("use_stratified_sampler", True)
        if use_stratified and len(train_dataset) > 0:
            train_batch_sampler = StratifiedBatchSampler(
                labels=train_dataset._labels,
                batch_size=self.config["batch_size"],
                drop_last=drop_last,
            )
            self.train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_batch_sampler,
                num_workers=self.config.get("num_workers", 4),
                pin_memory=self.config.get("pin_memory", True),
                collate_fn=train_dataset.collate_fn,
            )
            print(f"  Using StratifiedBatchSampler for balanced batches")
        else:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=self.config.get("num_workers", 4),
                pin_memory=self.config.get("pin_memory", True),
                collate_fn=train_dataset.collate_fn,
                drop_last=drop_last,
            )
        
        # Validation dataset (same as train for now)
        val_dataset = WeldingDataset(
            data_root=DATA_ROOT,
            video_length=VIDEO_LENGTH,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            audio_duration=AUDIO_DURATION,
            sensor_length=SENSOR_LENGTH,
            image_size=IMAGE_SIZE,
            num_angles=IMAGE_NUM_ANGLES,
            dummy=self.use_dummy,
        )
        
        if self.debug:
            val_label_counts = Counter(val_dataset._labels)
            print(f"  Val label distribution: {dict(sorted(val_label_counts.items()))}")

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=self.config.get("pin_memory", True),
            collate_fn=val_dataset.collate_fn,
        )
        
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print()
        
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        print("=" * 70)
        print("INITIALIZING OPTIMIZER")
        print("=" * 70)
        
        optimizer_name = self.config.get("optimizer", "adamw").lower()
        lr = self.config["learning_rate"]
        weight_decay = self.config.get("weight_decay", 1e-4)
        
        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                **OPTIMIZER_CONFIGS["adam"]
            )
        elif optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                **OPTIMIZER_CONFIGS["adamw"]
            )
        elif optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                **OPTIMIZER_CONFIGS["sgd"]
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        print(f"  Optimizer: {optimizer_name}")
        print(f"  Learning rate: {lr}")
        print(f"  Weight decay: {weight_decay}")
        
        # Setup scheduler
        scheduler_name = self.config.get("lr_scheduler", "cosine").lower()
        
        if scheduler_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["num_epochs"],
                eta_min=self.config.get("min_lr", 1e-6),
            )
        elif scheduler_name == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                **SCHEDULER_CONFIGS["step"]
            )
        elif scheduler_name == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                **SCHEDULER_CONFIGS["plateau"]
            )
        elif scheduler_name == "none":
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        if self.scheduler:
            print(f"  Scheduler: {scheduler_name}")
        print()
        
    def _setup_loss(self):
        """Setup loss function."""
        print("=" * 70)
        print("INITIALIZING LOSS")
        print("=" * 70)
        
        loss_type = self.config.get("loss_type", "supcon")
        temperature = self.config.get("temperature", 0.07)
        
        if loss_type == "supcon":
            self.criterion = SupConLoss(temperature=temperature)
            print(f"  Loss: Supervised Contrastive")
            print(f"  Temperature: {temperature}")
        elif loss_type == "combined":
            self.criterion = CombinedLoss(
                use_ce=self.config.get("use_ce", False),
                ce_weight=self.config.get("ce_weight", 0.0),
                supcon_weight=self.config.get("supcon_weight", 1.0),
                temperature=temperature,
                num_classes=NUM_CLASSES,
            )
            print(f"  Loss: Combined (SupCon + CE)")
            print(f"  Temperature: {temperature}")
            print(f"  CE weight: {self.config.get('ce_weight', 0.0)}")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        print()
        
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)

            # Lightweight debug: print labels stats for first batch when enabled
            if getattr(self, 'debug', False) and batch_idx == 0:
                try:
                    lbls = batch.get('label')
                    if isinstance(lbls, torch.Tensor):
                        unique, counts = torch.unique(lbls, return_counts=True)
                        print(f"[DEBUG] Batch labels - unique: {unique.tolist()}, counts: {counts.tolist()}")
                    else:
                        print(f"[DEBUG] Labels (non-tensor): {lbls}")
                except Exception as e:
                    print(f"[DEBUG] Failed to inspect labels: {e}")
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    features = self.model(batch)
            else:
                features = self.model(batch)

            # Debug: inspect feature statistics for first batch
            if getattr(self, 'debug', False) and batch_idx == 0:
                try:
                    if isinstance(features, torch.Tensor):
                        if features.shape[0] < 2:
                            print("[DEBUG] Batch has fewer than 2 samples; SupCon loss will be zero. Consider increasing batch size or ensuring stratified sampling.")
                        f_mean = float(features.mean().detach())
                        f_std = float(features.std().detach())
                        f_min = float(features.min().detach())
                        f_max = float(features.max().detach())
                        any_nan = bool(torch.isnan(features).any())
                        any_inf = bool(torch.isinf(features).any())
                        print(f"[DEBUG] Features - mean: {f_mean:.6f}, std: {f_std:.6f}, "
                              f"min: {f_min:.6f}, max: {f_max:.6f}, nan: {any_nan}, inf: {any_inf}")
                        
                        # Pairwise cosine similarities (small sample)
                        if features.dim() == 2:
                            feats_norm = F.normalize(features, dim=1)
                            cos_sim = torch.matmul(feats_norm, feats_norm.T)
                            l2_dist = torch.cdist(features, features, p=2)
                            b = cos_sim.size(0)
                            display_n = min(5, b)
                            
                            print(f"[DEBUG] Pairwise cosine similarity (top {display_n}x{display_n}):")
                            cos_small = cos_sim[:display_n, :display_n].detach().cpu().numpy()
                            for row in cos_small:
                                print(f"        {' '.join(f'{v:7.4f}' for v in row)}")
                            
                            print(f"[DEBUG] Pairwise L2 distance (top {display_n}x{display_n}):")
                            l2_small = l2_dist[:display_n, :display_n].detach().cpu().numpy()
                            for row in l2_small:
                                print(f"        {' '.join(f'{v:7.3f}' for v in row)}")
                    else:
                        print(f"[DEBUG] Features not a tensor: {type(features)}")
                except Exception as e:
                    print(f"[DEBUG] Failed to inspect features: {e}")

            # Compute loss
            loss_output = self.criterion(features, batch["label"])
            
            # Handle both scalar and dict returns
            if isinstance(loss_output, dict):
                loss = loss_output['total']
                loss_value = loss.item()
            else:
                loss = loss_output
                loss_value = loss.item()
            
            # Debug: inspect loss computation for first batch
            if getattr(self, 'debug', False) and batch_idx == 0:
                try:
                    labels = batch['label']
                    temp = getattr(self.criterion, 'temperature', 0.07)
                    
                    # Reproduce SupCon computation on device
                    feats_norm = F.normalize(features.detach(), dim=1)
                    sim_matrix = torch.matmul(feats_norm, feats_norm.T) / temp
                    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
                    logits = sim_matrix - logits_max
                    
                    # Positive mask (same label, excluding diagonal)
                    labels_col = labels.contiguous().view(-1, 1)
                    mask_pos = torch.eq(labels_col, labels_col.T).float()
                    mask_diag = torch.eye(mask_pos.size(0), device=self.device)
                    mask_pos = mask_pos * (1 - mask_diag)
                    
                    mask_sum = mask_pos.sum(1)
                    exp_logits = torch.exp(logits) * (1 - mask_diag)
                    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
                    mean_log_prob_pos = (mask_pos * log_prob).sum(1) / torch.clamp(mask_sum, min=1.0)
                    
                    print(f"[DEBUG] SupCon internals:")
                    print(f"        Temperature: {temp}")
                    print(f"        Pos pairs per sample: {mask_sum.cpu().numpy()}")
                    print(f"        Mean log prob (first 5): {mean_log_prob_pos[:5].cpu().numpy()}")
                    if float(mask_sum.max().item()) < 1.0:
                        print("        ⚠️ No positive pairs in this batch; SupCon loss will be ~0. Increase batch size or adjust sampler.")
                    print(f"        Final loss value: {loss_value:.6f}")
                except Exception as e:
                    print(f"[DEBUG] Failed to inspect loss: {e}")
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.config.get("gradient_clip", 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["gradient_clip"]
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.get("gradient_clip", 0) > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["gradient_clip"]
                    )
                self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss_value
            self.global_step += 1
            
            # Logging
            if (batch_idx + 1) % self.config.get("log_interval", 10) == 0:
                lr = self.optimizer.param_groups[0]['lr']
                avg_loss = epoch_loss / (batch_idx + 1)
                
                print(f"  [{epoch:3d}][{batch_idx+1:3d}/{len(self.train_loader):3d}] "
                      f"Loss: {loss_value:.4f} | Avg: {avg_loss:.4f} | LR: {lr:.2e}")
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(self.train_loader)
        
        return {
            "loss": avg_loss,
            "time": epoch_time,
            "lr": self.optimizer.param_groups[0]['lr'],
        }
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate model."""
        self.model.eval()
        
        val_loss = 0.0
        
        for batch in self.val_loader:
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass
            features = self.model(batch)
            loss_output = self.criterion(features, batch["label"])
            
            # Handle both scalar and dict returns
            if isinstance(loss_output, dict):
                loss_value = loss_output['total'].item()
            else:
                loss_value = loss_output.item()
            
            val_loss += loss_value
        
        avg_loss = val_loss / len(self.val_loader)
        
        return {"loss": avg_loss}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save latest
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"  ✅ Saved best model (epoch {epoch})")
        
        # Save periodic
        if (epoch + 1) % self.config.get("save_interval", 5) == 0:
            epoch_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pth"
            torch.save(checkpoint, epoch_path)
    
    def train(self):
        """Main training loop."""
        print("=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"  Epochs: {self.config['num_epochs']}")
        print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        for epoch in range(self.config["num_epochs"]):
            self.current_epoch = epoch
            
            print(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            print("-" * 70)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_log.append(train_metrics)
            
            # Validate
            if (epoch + 1) % self.config.get("val_interval", 1) == 0:
                val_metrics = self.validate(epoch)
                self.val_log.append(val_metrics)
                
                print(f"  Validation Loss: {val_metrics['loss']:.4f}")
                
                # Check if best
                is_best = val_metrics["loss"] < self.best_metric
                if is_best:
                    self.best_metric = val_metrics["loss"]
                
                # Save checkpoint
                self.save_checkpoint(epoch, is_best=is_best)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()
            
            print(f"  Epoch time: {train_metrics['time']:.1f}s")
            print()
        
        # Save final logs
        self._save_logs()
        
        print("=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Best val loss: {self.best_metric:.4f}")
        print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def _save_logs(self):
        """Save training logs."""
        log_data = {
            "train": self.train_log,
            "val": self.val_log,
            "config": self.config,
            "best_metric": self.best_metric,
        }
        
        log_path = self.log_dir / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"  Logs saved to: {log_path}")


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description='Train QuadModal model')
    parser.add_argument('--batch-size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--num-epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--device', type=str, help='Device to use, e.g. "cuda" or "cpu"')
    parser.add_argument('--mixed-precision', action='store_true', help='Enable AMP')
    parser.add_argument('--quick-test', action='store_true', help='Run quick smoke test (dummy data, small config)')
    parser.add_argument('--use-dummy', action='store_true', help='Use dummy datasets and encoders')
    parser.add_argument('--debug', action='store_true', help='Enable lightweight debug logs for first batch')
    args = parser.parse_args()

    # Base config from file
    config = TRAIN_CONFIG.copy()
    
    # Default to CUDA if available (server environment)
    if 'device' not in config or config['device'] == 'cuda':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # If quick-test: sensible small overrides for fast smoke runs
    if args.quick_test:
        config['num_epochs'] = 3
        config['batch_size'] = 4
        config['log_interval'] = 1
        config['val_interval'] = 1
        config['save_interval'] = 1
        use_dummy = True
    else:
        use_dummy = bool(args.use_dummy)

    # Command-line overrides
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs
    if args.device is not None:
        config['device'] = args.device
    if args.mixed_precision:
        config['mixed_precision'] = True

    # Create trainer
    trainer = Trainer(config, use_dummy=use_dummy)
    # Optional lightweight debug info printed for first batch
    trainer.debug = bool(getattr(args, 'debug', False))

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
