"""
Train a simple Video Autoencoder for Convexity detection.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import WeldingDataset
from src.models.video_ae import SimpleVideoAE, VideoFeatureExtractor
from configs.dataset_config import (
    DATA_ROOT, MANIFEST_PATH, VIDEO_LENGTH, IMAGE_SIZE
)
from configs.video_ae_config import VIDEO_AE_CONFIG

def main():
    # Configuration
    cfg = VIDEO_AE_CONFIG
    BATCH_SIZE = cfg["batch_size"]
    NUM_EPOCHS = cfg["num_epochs"]
    LEARNING_RATE = cfg["learning_rate"]
    DEVICE = cfg["device"] if torch.cuda.is_available() else "cpu"
    SAVE_PATH = cfg["save_path"]
    
    # Initialize WandB
    if cfg.get("use_wandb", False):
        wandb.init(
            project=cfg.get("wandb_project", "weld-anomaly-detection"),
            entity=cfg.get("wandb_entity", None),
            name=cfg.get("wandb_run_name", "video-ae-training"),
            config=cfg
        )

    print(f"Training Video AE on {DEVICE}")
    
    # Create checkpoints directory
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    # Dataset
    # We use 'train' split for training
    dataset = WeldingDataset(
        root_dir=DATA_ROOT,
        mode="real",
        split="train",
        manifest_path=MANIFEST_PATH,
        num_frames=VIDEO_LENGTH,
        image_size=IMAGE_SIZE, # DINOv2 usually likes 224
        augment=False # No augmentation for feature extraction usually, or maybe yes? 
                      # User said "Use existing WeldDataset". 
                      # For AE training, maybe we want clean features. 
                      # Let's keep augment=False for stability of features.
    )
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Models
    # Feature Extractor (Frozen)
    feature_extractor = VideoFeatureExtractor(model_name=cfg["model_name"], device=DEVICE)
    
    # Autoencoder (Trainable)
    model = SimpleVideoAE(input_dim=cfg["input_dim"]).to(DEVICE)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    # Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in pbar:
            # Get video data: (B, T, 3, H, W)
            video = batch['video'].to(DEVICE)
            
            # Extract features: (B, 768)
            with torch.no_grad():
                features = feature_extractor(video)
            
            # Forward pass
            reconstructed = model(features)
            
            # Compute loss
            loss = criterion(reconstructed, features)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
            
            if cfg.get("use_wandb", False):
                wandb.log({
                    "train/loss": loss.item(),
                    "epoch": epoch + 1,
                    "batch": num_batches
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
        
        if cfg.get("use_wandb", False):
            wandb.log({
                "train/epoch_loss": avg_loss,
                "epoch": epoch + 1
            })
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved best model to {SAVE_PATH}")

    if cfg.get("use_wandb", False):
        wandb.finish()

if __name__ == "__main__":
    main()
