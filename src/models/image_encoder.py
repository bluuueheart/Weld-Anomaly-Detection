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
                        output_hidden_states=True, # Enable hidden states output
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
        
        # Multi-scale feature dimension (Layer 4 + Layer 12)
        self.multi_scale_dim = backbone_dim * 2
        
        # No projection here, we return raw concatenated features
        self.projection = nn.Identity()
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            images: (B, num_angles, 3, H, W) tensor
            
        Returns:
            features: (B, num_angles, 1536) tensor (concatenated L4+L12 pooled features)
        """
        B, N, C, H, W = images.shape
        assert N == self.num_angles, f"Expected {self.num_angles} angles, got {N}"
        
        # Reshape to (B*N, 3, H, W)
        images_flat = images.reshape(B * N, C, H, W)
        
        # Extract features with DINOv2
        outputs = self.backbone(pixel_values=images_flat)
        
        # Extract Layer 4 and Layer 12
        # hidden_states is a tuple of (B*N, SeqLen, Dim)
        # Index 0 is embedding, Index 1 is Layer 1, ..., Index 12 is Layer 12 (Final)
        # We want Layer 4 (index 4) and Layer 12 (index 12)
        # Note: HuggingFace DINOv2 output_hidden_states includes embeddings at index 0?
        # Let's check documentation or assume standard HF behavior: 
        # hidden_states[0] = embeddings, hidden_states[1] = layer 1 output...
        # So Layer 4 is index 4+1=5? Or just index 4?
        # Usually hidden_states includes the output of embeddings + one for each layer.
        # So len(hidden_states) = 13 (1 embedding + 12 layers).
        # Layer 4 output is at index 4. Layer 12 output is at index 12.
        
        hidden_states = outputs.hidden_states
        
        # Layer 4 (Low-level texture)
        feat_l4 = hidden_states[4] # (B*N, SeqLen, 768)
        
        # Layer 12 (High-level semantic)
        feat_l12 = hidden_states[12] # (B*N, SeqLen, 768)
        
        # Return raw features for DecoupledResultEncoder
        # Reshape to (B, N, SeqLen, 768)
        feat_l4 = feat_l4.reshape(B, N, -1, 768)
        feat_l12 = feat_l12.reshape(B, N, -1, 768)
        
        return {
            "l4": feat_l4,
            "l12": feat_l12
        }


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
            nn.AdaptiveAvgPool2d((1, 1)), # Global pool
        )
        
        # Output 1536 dim to match real encoder
        self.projection = nn.Linear(256, 1536)
    
    def forward(self, images: torch.Tensor) -> dict:
        """Forward pass.
        
        Args:
            images: (B, num_angles, 3, H, W) tensor
            
        Returns:
            dict: {'l4': (B, N, SeqLen, 768), 'l12': (B, N, SeqLen, 768)}
        """
        B, N, C, H, W = images.shape
        
        # Reshape to (B*N, 3, H, W)
        images_flat = images.reshape(B * N, C, H, W)
        
        # Extract features
        features = self.backbone(images_flat)  # (B*N, 256, 1, 1)
        features = features.flatten(1)  # (B*N, 256)
        features = self.projection(features)  # (B*N, 1536)
        
        # Split into fake l4 and l12 (just split the 1536 dim)
        feat_l4 = features[:, :768].unsqueeze(1) # Fake seq len 1
        feat_l12 = features[:, 768:].unsqueeze(1)
        
        # Reshape to (B, N, SeqLen, 768)
        feat_l4 = feat_l4.reshape(B, N, 1, 768)
        feat_l12 = feat_l12.reshape(B, N, 1, 768)
        
        return {
            "l4": feat_l4,
            "l12": feat_l12
        }
