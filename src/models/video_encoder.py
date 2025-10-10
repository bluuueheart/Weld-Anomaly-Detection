"""Video encoder using V-JEPA."""
from typing import Optional
import torch
import torch.nn as nn


class VideoEncoder(nn.Module):
    """Video encoder using V-JEPA (facebook/vjepa2-vitl-fpc64-256).
    
    Input: (batch, num_frames, channels, height, width)
    Output: (batch, seq_len, embed_dim)
    """
    
    def __init__(
        self,
        model_path: str = "models/vjepa2-vitl-fpc64-256",
        embed_dim: int = 1024,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize video encoder.
        
        Args:
            model_path: Path to pretrained V-JEPA model
            embed_dim: Output embedding dimension
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        try:
            from transformers import AutoModel
        except ImportError:
            raise RuntimeError("transformers is required for VideoEncoder")
        
        # Load pretrained V-JEPA model
        try:
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            # Fallback to online loading
            self.model = AutoModel.from_pretrained(
                "facebook/vjepa2-vitl-fpc64-256",
                trust_remote_code=True,
            )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Get model output dimension
        self.model_dim = self.model.config.hidden_size if hasattr(self.model.config, 'hidden_size') else 1024
        
        # Projection layer if needed
        if self.model_dim != embed_dim:
            self.projection = nn.Linear(self.model_dim, embed_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Video tensor (batch, num_frames, channels, height, width)
        
        Returns:
            features: (batch, seq_len, embed_dim)
        """
        batch_size, num_frames, channels, height, width = x.shape
        
        # V-JEPA expects (batch, channels, num_frames, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        
        # Get features from V-JEPA
        outputs = self.model(x)
        
        # Extract features (model-dependent, may need adjustment)
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            features = outputs[0]
        else:
            features = outputs
        
        # Project to target dimension
        features = self.projection(features)
        
        return features


class DummyVideoEncoder(nn.Module):
    """Dummy video encoder for testing without pretrained model.
    
    Input: (batch, num_frames, channels, height, width)
    Output: (batch, seq_len, embed_dim)
    """
    
    def __init__(self, embed_dim: int = 1024, seq_len: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        
        # Simple conv + pool to reduce spatial dimensions
        self.conv = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((seq_len, 1, 1)),
        )
        self.proj = nn.Linear(128, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = self.conv(x)  # (B, 128, seq_len, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B, 128, seq_len)
        x = x.permute(0, 2, 1)  # (B, seq_len, 128)
        x = self.proj(x)  # (B, seq_len, embed_dim)
        return x


__all__ = ["VideoEncoder", "DummyVideoEncoder"]
