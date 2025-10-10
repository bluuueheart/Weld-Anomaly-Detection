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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import create_quadmodal_model
from src.dataset import WeldingDataset
from src.losses import SupConLoss, CombinedLoss
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
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=self.config.get("pin_memory", True),
            collate_fn=train_dataset.collate_fn,
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
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    features = self.model(batch)
                    loss = self.criterion(features, batch["label"])
            else:
                features = self.model(batch)
                loss = self.criterion(features, batch["label"])
            
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
            epoch_loss += loss.item()
            self.global_step += 1
            
            # Logging
            if (batch_idx + 1) % self.config.get("log_interval", 10) == 0:
                lr = self.optimizer.param_groups[0]['lr']
                avg_loss = epoch_loss / (batch_idx + 1)
                
                print(f"  [{epoch:3d}][{batch_idx+1:3d}/{len(self.train_loader):3d}] "
                      f"Loss: {loss.item():.4f} | Avg: {avg_loss:.4f} | LR: {lr:.2e}")
        
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
            loss = self.criterion(features, batch["label"])
            
            val_loss += loss.item()
        
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
    # Use dummy mode for testing
    use_dummy = True
    
    # Override config for quick test
    config = TRAIN_CONFIG.copy()
    config["num_epochs"] = 3
    config["batch_size"] = 4
    config["log_interval"] = 1
    config["val_interval"] = 1
    config["save_interval"] = 1
    config["device"] = "cpu"  # Use CPU for testing
    
    # Create trainer
    trainer = Trainer(config, use_dummy=use_dummy)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
