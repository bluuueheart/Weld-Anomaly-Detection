"""Audio encoder using AST (Audio Spectrogram Transformer)."""
from typing import Optional
import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """Audio encoder using AST (MIT/ast-finetuned-audioset-14-14-0.443).
    
    Input: (batch, 1, n_mels, time_frames) - mel spectrogram
    Output: (batch, seq_len, embed_dim)
    """
    
    def __init__(
        self,
        model_path: str = "models/ast-finetuned-audioset-14-14-0.443",
        embed_dim: int = 768,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize audio encoder.
        
        Args:
            model_path: Path to pretrained AST model
            embed_dim: Output embedding dimension
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        try:
            from transformers import AutoModel
        except ImportError:
            raise RuntimeError("transformers is required for AudioEncoder")
        
        # Load pretrained AST model
        try:
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            # Fallback to online loading
            self.model = AutoModel.from_pretrained(
                "MIT/ast-finetuned-audioset-14-14-0.443",
                trust_remote_code=True,
            )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Get model output dimension
        self.model_dim = self.model.config.hidden_size if hasattr(self.model.config, 'hidden_size') else 768
        
        # Projection layer if needed
        if self.model_dim != embed_dim:
            self.projection = nn.Linear(self.model_dim, embed_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Mel spectrogram (batch, 1, n_mels, time_frames)
        
        Returns:
            features: (batch, seq_len, embed_dim)
        """
        # AST expects (batch, n_mels, time_frames)
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        
        # Get features from AST
        outputs = self.model(x)
        
        # Extract features
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            features = outputs[0]
        else:
            features = outputs
        
        # Project to target dimension
        features = self.projection(features)
        
        return features


class DummyAudioEncoder(nn.Module):
    """Dummy audio encoder for testing without pretrained model.
    
    Input: (batch, 1, n_mels, time_frames)
    Output: (batch, seq_len, embed_dim)
    """
    
    def __init__(self, embed_dim: int = 768, seq_len: int = 32) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        
        # Simple conv layers
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((seq_len, 1)),
        )
        self.proj = nn.Linear(128, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.size(1) == 1:
            pass  # Already (B, 1, H, W)
        elif x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, H, W)
        
        x = self.conv(x)  # (B, 128, seq_len, 1)
        x = x.squeeze(-1)  # (B, 128, seq_len)
        x = x.permute(0, 2, 1)  # (B, seq_len, 128)
        x = self.proj(x)  # (B, seq_len, embed_dim)
        return x


__all__ = ["AudioEncoder", "DummyAudioEncoder"]
