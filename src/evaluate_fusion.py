"""
Evaluate Fusion Strategy: Causal-FiLM + Video AE.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import WeldingDataset
from src.models.causal_film_model import create_causal_film_model
from src.models.video_ae import SimpleVideoAE, VideoFeatureExtractor
from configs.dataset_config import (
    DATA_ROOT, MANIFEST_PATH, VIDEO_LENGTH, IMAGE_SIZE,
    AUDIO_SAMPLE_RATE, AUDIO_DURATION, SENSOR_LENGTH, IMAGE_NUM_ANGLES
)
from configs.model_config import (
    VIDEO_ENCODER, IMAGE_ENCODER, AUDIO_ENCODER, SENSOR_ENCODER, FUSION
)
from configs.video_ae_config import FUSION_CONFIG, VIDEO_AE_CONFIG

def main():
    # Configuration
    cfg = FUSION_CONFIG
    BATCH_SIZE = cfg["batch_size"]
    DEVICE = cfg["device"] if torch.cuda.is_available() else "cpu"
    
    # Checkpoint Paths
    CHECKPOINT_A = cfg["checkpoint_a"] # Causal Model
    CHECKPOINT_B = cfg["checkpoint_b"] # Video AE
    
    print(f"Evaluating Fusion on {DEVICE}")
    print(f"Model A: {CHECKPOINT_A}")
    print(f"Model B: {CHECKPOINT_B}")
    
    # Dataset (Test Split)
    dataset = WeldingDataset(
        root_dir=DATA_ROOT,
        mode="real",
        split="test",
        manifest_path=MANIFEST_PATH,
        num_frames=VIDEO_LENGTH,
        image_size=IMAGE_SIZE,
        augment=False
    )
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # --- Load Model A (Causal FiLM) ---
    print("Loading Model A...")
    model_config = {
        "VIDEO_ENCODER": VIDEO_ENCODER,
        "IMAGE_ENCODER": IMAGE_ENCODER,
        "AUDIO_ENCODER": AUDIO_ENCODER,
        "SENSOR_ENCODER": SENSOR_ENCODER,
        "FUSION": FUSION,
    }
    model_a = create_causal_film_model(model_config, use_dummy=False)
    
    if os.path.exists(CHECKPOINT_A):
        # Set weights_only=False to handle numpy scalars in checkpoints (PyTorch 2.6+ default change)
        checkpoint_a = torch.load(CHECKPOINT_A, map_location=DEVICE, weights_only=False)
        # Handle if checkpoint is full dict or just state_dict
        if "model_state_dict" in checkpoint_a:
            model_a.load_state_dict(checkpoint_a["model_state_dict"])
        else:
            model_a.load_state_dict(checkpoint_a)
    else:
        print(f"Warning: Checkpoint A not found at {CHECKPOINT_A}. Using random weights (for testing flow).")
    
    model_a.to(DEVICE)
    model_a.eval()
    
    # --- Load Model B (Video AE) ---
    print("Loading Model B...")
    feature_extractor_b = VideoFeatureExtractor(model_name=VIDEO_AE_CONFIG["model_name"], device=DEVICE)
    model_b = SimpleVideoAE(input_dim=VIDEO_AE_CONFIG["input_dim"])
    
    if os.path.exists(CHECKPOINT_B):
        model_b.load_state_dict(torch.load(CHECKPOINT_B, map_location=DEVICE, weights_only=False))
    else:
        print(f"Warning: Checkpoint B not found at {CHECKPOINT_B}. Using random weights.")
        
    model_b.to(DEVICE)
    model_b.eval()
    
    # --- Inference ---
    scores_a = []
    scores_b = []
    labels = []
    
    print("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move data to device
            video = batch['video'].to(DEVICE)
            audio = batch['audio'].to(DEVICE)
            sensor = batch['sensor'].to(DEVICE)
            images = batch['post_weld_images'].to(DEVICE)
            label = batch['label'].to(DEVICE)
            
            # --- Model A Inference ---
            batch_input = {
                'video': video,
                'audio': audio,
                'post_weld_images': images,
                'sensor': sensor
            }
            out_a = model_a(batch_input)
            # Score A: L1 distance
            # Shape: (B, 128)
            z_result = out_a['Z_result']
            z_pred = out_a['Z_result_pred']
            score_a_batch = torch.mean(torch.abs(z_result - z_pred), dim=1) # (B,)
            
            # --- Model B Inference ---
            # Extract features
            features_b = feature_extractor_b(video) # (B, 768)
            recon_b = model_b(features_b)
            # Score B: MSE distance
            # MSE = mean((x-y)^2) per sample
            score_b_batch = torch.mean((features_b - recon_b) ** 2, dim=1) # (B,)
            
            scores_a.extend(score_a_batch.cpu().numpy())
            scores_b.extend(score_b_batch.cpu().numpy())
            labels.extend(label.cpu().numpy())
            
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    labels = np.array(labels)
    
    # Binary labels (0 is normal, >0 is anomaly)
    binary_labels = (labels > 0).astype(int)
    
    # --- Post-Processing ---
    # Compute Mean/Std
    mean_a = np.mean(scores_a)
    std_a = np.std(scores_a)
    
    mean_b = np.mean(scores_b)
    std_b = np.std(scores_b)
    
    print(f"Model A Stats: Mean={mean_a:.4f}, Std={std_a:.4f}")
    print(f"Model B Stats: Mean={mean_b:.4f}, Std={std_b:.4f}")
    
    # Normalize (Z-score)
    # Avoid division by zero
    std_a = std_a if std_a > 1e-6 else 1.0
    std_b = std_b if std_b > 1e-6 else 1.0
    
    z_a = (scores_a - mean_a) / std_a
    z_b = (scores_b - mean_b) / std_b
    
    # Final Score
    final_scores = z_a + z_b
    
    # --- Metrics ---
    try:
        auroc = roc_auc_score(binary_labels, final_scores)
        print(f"\nFinal I-AUROC: {auroc:.4f}")
        
        # Individual AUROCs for comparison
        auroc_a = roc_auc_score(binary_labels, scores_a)
        auroc_b = roc_auc_score(binary_labels, scores_b)
        print(f"Model A I-AUROC: {auroc_a:.4f}")
        print(f"Model B I-AUROC: {auroc_b:.4f}")
        
    except ValueError as e:
        print(f"Error computing AUROC: {e}")
        print("Ensure test set has both normal and anomaly samples.")

if __name__ == "__main__":
    main()
