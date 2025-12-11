"""Training script for Late Fusion baseline models.

Trains audio and video auto-encoders separately following the paper's specifications.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from baselines.Late_Fusion.models import AudioAutoEncoder, VideoAutoEncoder
from baselines.Late_Fusion.config import AUDIO_CONFIG, VIDEO_CONFIG, TRAIN_CONFIG
from baselines.Late_Fusion.utils import compute_stft, EarlyStopping
from src.dataset import WeldingDataset


def train_audio_autoencoder(
    model: AudioAutoEncoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: str = "cuda",
    save_dir: str = "checkpoints",
):
    """Train audio auto-encoder.
    
    Args:
        model: Audio auto-encoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on
        save_dir: Directory to save checkpoints
    """
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()
    
    # One-cycle learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        epochs=config["num_epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("=" * 70)
    print("TRAINING AUDIO AUTO-ENCODER")
    print("=" * 70)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Device: {device}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print()
    
    for epoch in range(config["num_epochs"]):
        # Training
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            audio = batch["audio"].to(device)  # (batch, 1, n_mels, time)
            
            # Convert mel-spectrogram to STFT-like representation
            # For now, we'll use the mel-spectrogram directly
            # In practice, you'd compute STFT from raw audio
            audio_input = audio.squeeze(1)  # (batch, n_mels, time)
            
            # Forward pass
            reconstruction = model(audio_input)
            loss = criterion(reconstruction, audio_input)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{config['num_epochs']}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.6f} "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device)
                audio_input = audio.squeeze(1)
                
                reconstruction = model(audio_input)
                loss = criterion(reconstruction, audio_input)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] "
              f"Train Loss: {avg_train_loss:.6f} "
              f"Val Loss: {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "audio_autoencoder_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "config": config,
            }, save_path)
            print(f"  → Saved best model to {save_path}")
    
    # Save final model
    final_path = os.path.join(save_dir, "audio_autoencoder_final.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "config": config,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, final_path)
    print(f"\nTraining completed. Final model saved to {final_path}")
    
    return model, train_losses, val_losses


def train_video_autoencoder(
    model: VideoAutoEncoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: str = "cuda",
    save_dir: str = "checkpoints",
):
    """Train video auto-encoder (Stage 2 only, SlowFast frozen).
    
    Args:
        model: Video auto-encoder model
        train_loader: Training data loader (features pre-extracted)
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on
        save_dir: Directory to save checkpoints
    """
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"]
    )
    criterion = nn.MSELoss()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.get("early_stopping_patience", 50))
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("=" * 70)
    print("TRAINING VIDEO AUTO-ENCODER (Stage 2)")
    print("=" * 70)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Device: {device}")
    print(f"Max epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print()
    
    for epoch in range(config["num_epochs"]):
        # Training
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            video = batch["video"].to(device)  # (batch, num_frames, C, H, W)
            
            # Forward pass (features will be extracted inside model)
            reconstruction = model(video)
            
            # Get features for loss computation
            with torch.no_grad():
                features = model.extract_features(video)
            
            loss = criterion(reconstruction, features)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{config['num_epochs']}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.6f}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                video = batch["video"].to(device)
                
                reconstruction = model(video)
                features = model.extract_features(video)
                loss = criterion(reconstruction, features)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] "
              f"Train Loss: {avg_train_loss:.6f} "
              f"Val Loss: {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "video_autoencoder_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "config": config,
            }, save_path)
            print(f"  → Saved best model to {save_path}")
        
        # Check early stopping
        if early_stopping(avg_val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Save final model
    final_path = os.path.join(save_dir, "video_autoencoder_final.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "config": config,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, final_path)
    print(f"\nTraining completed. Final model saved to {final_path}")
    
    return model, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description="Train Late Fusion baseline models")
    parser.add_argument("--modality", type=str, choices=["audio", "video", "both"], default="both",
                        help="Which modality to train")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to dataset root")
    parser.add_argument("--manifest", type=str, default="configs/manifest.csv",
                        help="Path to manifest file")
    parser.add_argument("--save_dir", type=str, default="baselines/Late_Fusion/checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on")
    parser.add_argument("--dummy", action="store_true",
                        help="Use dummy data for testing")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(TRAIN_CONFIG["seed"])
    np.random.seed(TRAIN_CONFIG["seed"])
    
    # Create datasets
    print("Loading datasets...")
    
    # Calculate n_bins for STFT
    n_bins = AUDIO_CONFIG["n_fft"] // 2 + 1
    
    train_dataset = WeldingDataset(
        root_dir=args.data_root,
        mode="dummy" if args.dummy else "real",
        split="train",
        manifest_path=args.manifest,
        audio_type="stft",
        audio_sr=AUDIO_CONFIG["sample_rate"],
        n_fft=AUDIO_CONFIG["n_fft"],
        hop_length=AUDIO_CONFIG["hop_length"],
        audio_frames=1024,  # Will be determined by audio length
    )
    
    val_dataset = WeldingDataset(
        root_dir=args.data_root,
        mode="dummy" if args.dummy else "real",
        split="test",  # Use test split (consistent with main SOTA model)
        manifest_path=args.manifest,
        audio_type="stft",
        audio_sr=AUDIO_CONFIG["sample_rate"],
        n_fft=AUDIO_CONFIG["n_fft"],
        hop_length=AUDIO_CONFIG["hop_length"],
        audio_frames=1024,  # Will be determined by audio length
    )
    
    # Train audio model
    if args.modality in ["audio", "both"]:
        print("\n" + "=" * 70)
        print("AUDIO AUTO-ENCODER")
        print("=" * 70)
        
        # Calculate n_bins
        n_bins = AUDIO_CONFIG["n_fft"] // 2 + 1
        
        audio_model = AudioAutoEncoder(
            n_bins=n_bins,
            bottleneck_dim=AUDIO_CONFIG["bottleneck_dim"],
            hidden_channels=AUDIO_CONFIG["hidden_channels"],
            num_conv_layers=AUDIO_CONFIG["num_conv_layers"],
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=AUDIO_CONFIG["batch_size"],
            shuffle=True,
            num_workers=TRAIN_CONFIG["num_workers"],
            pin_memory=TRAIN_CONFIG["pin_memory"],
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=AUDIO_CONFIG["batch_size"],
            shuffle=False,
            num_workers=TRAIN_CONFIG["num_workers"],
            pin_memory=TRAIN_CONFIG["pin_memory"],
        )
        
        audio_model, audio_train_losses, audio_val_losses = train_audio_autoencoder(
            audio_model,
            train_loader,
            val_loader,
            AUDIO_CONFIG,
            device=args.device,
            save_dir=args.save_dir,
        )
    
    # Train video model
    if args.modality in ["video", "both"]:
        print("\n" + "=" * 70)
        print("VIDEO AUTO-ENCODER")
        print("=" * 70)
        
        video_model = VideoAutoEncoder(
            feature_dim=VIDEO_CONFIG["feature_dim"],
            encoder_layers=VIDEO_CONFIG["encoder_layers"],
            decoder_layers=VIDEO_CONFIG["decoder_layers"],
            dropout=VIDEO_CONFIG["dropout"],
            slowfast_model=None,  # Will be loaded separately if needed
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=VIDEO_CONFIG["batch_size"],
            shuffle=True,
            num_workers=TRAIN_CONFIG["num_workers"],
            pin_memory=TRAIN_CONFIG["pin_memory"],
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=VIDEO_CONFIG["batch_size"],
            shuffle=False,
            num_workers=TRAIN_CONFIG["num_workers"],
            pin_memory=TRAIN_CONFIG["pin_memory"],
        )
        
        video_model, video_train_losses, video_val_losses = train_video_autoencoder(
            video_model,
            train_loader,
            val_loader,
            VIDEO_CONFIG,
            device=args.device,
            save_dir=args.save_dir,
        )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
