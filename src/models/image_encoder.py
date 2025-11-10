"""Image encoder for post-weld multi-angle images using DINOv2."""
import torch
import torch.nn as nn
from typing import Optional


class ImageEncoder(nn.Module):
    """Encode multi-angle post-weld images using DINOv2.
    
    Architecture:
    - Loads pretrained facebook/dinov2-base model
    - Processes (B, num_angles, 3, H, W) input
    - Aggregates features across angles
    - Outputs (B, seq_len, embed_dim) features
    
    Args:
        model_name: HuggingFace model name (default: facebook/dinov2-base)
        embed_dim: Output embedding dimension (default: 768, DINOv2-base native)
        num_angles: Number of angles per sample (default: 5)
        aggregation: How to aggregate multi-angle features ('mean', 'max', 'concat')
        freeze_backbone: Whether to freeze pretrained weights
        local_model_path: Local path to model if available
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        embed_dim: int = 768,
        num_angles: int = 5,
        aggregation: str = "mean",
        freeze_backbone: bool = False,
        local_model_path: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.num_angles = num_angles
        self.aggregation = aggregation
        
        # Load pretrained DINOv2
        try:
            from transformers import AutoModel
            import os

            model_path = local_model_path if local_model_path else model_name
            model_path_resolved = model_path
            if not os.path.isabs(model_path_resolved):
                model_path_resolved = os.path.join(os.getcwd(), model_path_resolved)

            if os.path.exists(model_path_resolved):
                try:
                    self.backbone = AutoModel.from_pretrained(
                        model_path_resolved,
                        local_files_only=True,
                        trust_remote_code=True,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load local DINOv2 model at '{model_path_resolved}': {e}.\n"
                        "Ensure the folder contains 'config.json' and model weights (pytorch_model.bin or model.safetensors), and required packages are installed."
                    )
            else:
                raise FileNotFoundError(
                    f"Local DINOv2 model directory not found: '{model_path_resolved}'.\n"
                    "Please place the pretrained model in that path before running."
                )
        except ImportError as e:
            raise RuntimeError(
                "transformers is required for ImageEncoder. "
                "Install with: pip install transformers"
            ) from e
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get backbone output dimension
        backbone_dim = self.backbone.config.hidden_size
        
        # Projection layer with moderate dropout (trainable even when backbone frozen)
        if backbone_dim != embed_dim:
            self.projection = nn.Sequential(
                nn.Dropout(p=0.2),  # Moderate dropout
                nn.Linear(backbone_dim, embed_dim),
            )
        else:
            self.projection = nn.Dropout(p=0.2)  # Dropout for Identity case
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            images: (B, num_angles, 3, H, W) tensor
            
        Returns:
            features: (B, seq_len, embed_dim) tensor
        """
        B, N, C, H, W = images.shape
        assert N == self.num_angles, f"Expected {self.num_angles} angles, got {N}"
        
        # Reshape to (B*N, 3, H, W)
        images_flat = images.reshape(B * N, C, H, W)
        
        # Extract features with DINOv2
        outputs = self.backbone(pixel_values=images_flat)
        # Get [CLS] token or patch tokens
        # DINOv2 outputs: last_hidden_state shape (B*N, num_patches+1, dim)
        features = outputs.last_hidden_state  # (B*N, seq_len, backbone_dim)
        
        # Project to target dimension
        features = self.projection(features)  # (B*N, seq_len, embed_dim)
        
        # Reshape back to (B, N, seq_len, embed_dim)
        seq_len = features.shape[1]
        features = features.reshape(B, N, seq_len, self.embed_dim)
        
        # Aggregate across angles
        if self.aggregation == "mean":
            features = features.mean(dim=1)  # (B, seq_len, embed_dim)
        elif self.aggregation == "max":
            features = features.max(dim=1)[0]  # (B, seq_len, embed_dim)
        elif self.aggregation == "concat":
            # Concatenate along sequence dimension
            features = features.reshape(B, N * seq_len, self.embed_dim)
        elif self.aggregation == "none":
            # Do not aggregate across angles here; return per-view pooled features.
            # Pool over sequence dimension (take CLS token or mean over patches).
            # Prefer CLS token at index 0 if available.
            try:
                features = features[:, :, 0, :]
            except Exception:
                features = features.mean(dim=2)
            # Resulting shape: (B, N, embed_dim)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        return features


class DummyImageEncoder(nn.Module):
    """Dummy image encoder for testing without pretrained models.
    
    Uses a simple 2D CNN to process images.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_angles: int = 5,
        aggregation: str = "mean",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_angles = num_angles
        self.aggregation = aggregation
        
        # Simple CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        
        # Projection to embed_dim
        self.projection = nn.Linear(256 * 7 * 7, embed_dim)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            images: (B, num_angles, 3, H, W) tensor
            
        Returns:
            features: (B, 1, embed_dim) tensor (single token per sample)
        """
        B, N, C, H, W = images.shape
        
        # Reshape to (B*N, 3, H, W)
        images_flat = images.reshape(B * N, C, H, W)
        
        # Extract features
        features = self.backbone(images_flat)  # (B*N, 256, 7, 7)
        features = features.flatten(1)  # (B*N, 256*7*7)
        features = self.projection(features)  # (B*N, embed_dim)
        
        # Reshape to (B, N, embed_dim)
        features = features.reshape(B, N, self.embed_dim)
        
        # Aggregate across angles
        if self.aggregation == "mean":
            features = features.mean(dim=1, keepdim=True)  # (B, 1, embed_dim)
        elif self.aggregation == "max":
            features = features.max(dim=1, keepdim=True)[0]  # (B, 1, embed_dim)
        else:
            # concat: (B, N, embed_dim) -> keep as is
            pass
        
        return features
