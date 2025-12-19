"""Video Autoencoder model and feature extractor."""

import torch
import torch.nn as nn
import os
from typing import Optional

class SimpleVideoAE(nn.Module):
    """Simple MLP Autoencoder for video features.
    
    Structure: Linear(768->128) -> ReLU -> Linear(128->64) -> ReLU -> Linear(64->128) -> ReLU -> Linear(128->768).
    """
    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

class VideoFeatureExtractor(nn.Module):
    """Extracts video features using DINOv3.
    
    Processes video frames with DINOv3 and averages features over time.
    """
    def __init__(self, model_name: str = "/root/work/models/dinov3-vith16plus-pretrain-lvd1689m", device: str = "cuda"):
        super().__init__()
        self.device = device
        
        # Load pretrained DINOv2
        try:
            from transformers import AutoModel
            
            # Handle local path logic similar to ImageEncoder
            model_path_resolved = model_name
            if not os.path.isabs(model_path_resolved):
                model_path_resolved = os.path.join(os.getcwd(), model_path_resolved)

            if os.path.exists(model_path_resolved):
                try:
                    self.backbone = AutoModel.from_pretrained(
                        model_path_resolved,
                        local_files_only=True,
                        trust_remote_code=True,
                        output_hidden_states=True,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load local DINOv2 model at '{model_path_resolved}': {e}"
                    )
            else:
                # Fallback to huggingface hub if not found locally (or raise error if strict)
                # The user environment seems to prefer local, but we can try standard load if path doesn't exist
                # Assuming the user has the model downloaded as per instructions
                 raise FileNotFoundError(
                    f"Local DINOv2 model directory not found: '{model_path_resolved}'.\n"
                    "Please place the pretrained model in that path."
                )
                
        except ImportError:
            raise RuntimeError("transformers is required. Install with: pip install transformers")

        self.backbone.to(device)
        self.backbone.eval()
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (B, T, 3, H, W) tensor, normalized [0,1] or similar.
                   DINOv2 expects standard ImageNet normalization usually, 
                   but we'll assume the dataset provides reasonable input.
                   Transformers model usually handles pixel_values.
        Returns:
            features: (B, 768)
        """
        B, T, C, H, W = video.shape
        
        # Flatten to (B*T, C, H, W)
        video_flat = video.view(B * T, C, H, W).to(self.device)
        
        # DINOv2 forward
        with torch.no_grad():
            outputs = self.backbone(pixel_values=video_flat)
            # Extract Layer 12 (index 11) hidden states: (B*T, Seq, 768)
            hidden_states = outputs.hidden_states[11]
            
            # Mean over spatial tokens (dim 1) -> (B*T, 768)
            # hidden_states shape is (Batch, Sequence, Hidden)
            # We want to average over Sequence
            frame_features = hidden_states.mean(dim=1)
            
            # Reshape to (B, T, 768)
            frame_features = frame_features.view(B, T, -1)
            
            # Mean over time (dim 1) -> (B, 768)
            video_features = frame_features.mean(dim=1)
            
        return video_features
