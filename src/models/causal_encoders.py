"""
Causal Encoders for Causal-FiLM.

Implements:
- ProcessEncoder: Encodes "process" modalities (video + audio) via cross-attention
- ResultEncoder: Encodes "result" modality (post-weld images) via MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProcessEncoder(nn.Module):
    """
    Process Encoder using cross-attention between video and audio features.
    
    Architecture:
        - Video features (B, T_v, D) as Query
        - Audio features (B, T_a, D) as Key/Value
        - Cross-attention -> (B, T_v, D)
        - Mean pooling -> (B, D)
    
    This creates a causal dependency: audio modulates video representation.
    
    Args:
        d_model (int): Feature dimension (default: 128)
        num_heads (int): Number of attention heads (default: 4)
        num_layers (int): Number of cross-attention layers (default: 2)
        dropout (float): Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Cross-attention layers (video attends to audio)
        # Using TransformerDecoderLayer: tgt=video (Q), memory=audio (K,V)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_layers = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode process modalities via cross-attention.
        
        Args:
            video_features: Video features (B, T_v, d_model)
            audio_features: Audio features (B, T_a, d_model)
            
        Returns:
            Z_process: Process encoding (B, d_model)
        """
        # Cross-attention: video (Q) attends to audio (K, V)
        Z_process_tokens = self.cross_attn_layers(
            tgt=video_features,      # Query: video
            memory=audio_features,   # Key, Value: audio
        )  # (B, T_v, d_model)
        
        # Mean pooling over tokens
        Z_process = Z_process_tokens.mean(dim=1)  # (B, d_model)
        
        # Normalize
        Z_process = self.output_norm(Z_process)
        
        return Z_process


class ResultEncoder(nn.Module):
    """
    Decoupled Result Encoder (Dual-Stream).
    
    Splits processing into two streams:
    1. Texture Stream (Layer 4): Uses Top-K Pooling to capture small defects (Cracks).
    2. Structure Stream (Layer 12): Uses Mean Pooling to capture global defects (Warping).
    
    Args:
        d_model (int): Output feature dimension (default: 128)
        dropout (float): Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int = 128,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Stream 1: Texture (Layer 4)
        self.texture_projector = nn.Sequential(
            nn.Linear(768, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        
        # Stream 2: Structure (Layer 12)
        self.structure_projector = nn.Sequential(
            nn.Linear(768, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
    
    def forward(self, features_dict: dict) -> tuple:
        """
        Encode result modality (images).
        
        Args:
            features_dict: Dictionary containing:
                - 'l4': (B, N, SeqLen, 768)
                - 'l12': (B, N, SeqLen, 768)
            
        Returns:
            (z_texture, z_structure): Tuple of (B, d_model) tensors
        """
        feat_l4 = features_dict['l4']   # (B, N, SeqLen, 768)
        feat_l12 = features_dict['l12'] # (B, N, SeqLen, 768)
        
        B, N, S, C = feat_l4.shape
        
        # Flatten views and sequence for global pooling context
        # We want to find anomalies across all views
        flat_l4 = feat_l4.reshape(B, N * S, C)
        flat_l12 = feat_l12.reshape(B, N * S, C)
        
        # --- Stream 1: Texture (Focus on Cracks) ---
        # Use Top-K Mean Pooling (Top 16 patches across all views)
        # Exclude CLS token if it was included? DINOv2 CLS is usually index 0.
        # ImageEncoder returns raw hidden states. If it includes CLS, we should probably ignore it for texture.
        # Assuming ImageEncoder returns full sequence.
        # Let's assume we just take Top-K from whatever is there.
        
        # Top-K pooling
        k = min(16, flat_l4.size(1))
        topk_l4 = flat_l4.topk(k=k, dim=1, sorted=False)[0] # (B, k, 768)
        z_texture_raw = topk_l4.mean(dim=1) # (B, 768)
        
        z_texture = self.texture_projector(z_texture_raw) # (B, d_model)
        z_texture = F.normalize(z_texture, p=2, dim=-1)

        # --- Stream 2: Structure (Focus on Warping) ---
        # Use Standard Mean Pooling over all patches
        z_structure_raw = flat_l12.mean(dim=1) # (B, 768)
        
        z_structure = self.structure_projector(z_structure_raw) # (B, d_model)
        z_structure = F.normalize(z_structure, p=2, dim=-1)

        return z_texture, z_structure


class DummyProcessEncoder(nn.Module):
    """Lightweight dummy process encoder for testing."""
    
    def __init__(self, d_model: int = 128, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        """Simple concatenation and pooling."""
        # Concat and mean pool
        combined = torch.cat([video_features, audio_features], dim=1)
        pooled = combined.mean(dim=1)
        return pooled


class DummyResultEncoder(nn.Module):
    """Lightweight dummy result encoder for testing."""
    
    def __init__(self, d_model: int = 128, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.tex_proj = nn.Linear(768, d_model)
        self.struc_proj = nn.Linear(768, d_model)
    
    def forward(self, features_dict: dict) -> tuple:
        """Simple mean pooling and projection."""
        # Dummy implementation ignores input structure details
        # Just return random vectors or projected dummy inputs
        # Assuming inputs are tensors
        feat = features_dict['l4'] # (B, N, S, 768)
        pooled = feat.mean(dim=[1, 2]) # (B, 768)
        
        z_tex = F.normalize(self.tex_proj(pooled), p=2, dim=-1)
        z_struc = F.normalize(self.struc_proj(pooled), p=2, dim=-1)
        
        return z_tex, z_struc


class SpatialResultEncoder(nn.Module):
    """
    Spatial Result Encoder for post-weld images (Patch-based).
    
    Projects all patches to output dimension while maintaining spatial structure.
    """
    def __init__(self, input_dim=768, output_dim=128):
        super().__init__()
        # Use Linear to process sequence, weights shared across patches
        self.projector = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # x shape: (B, N_patches, input_dim)
        z_result = self.projector(x) 
        # z_result shape: (B, N_patches, output_dim)
        # Normalize for Cosine Distance
        z_result = F.normalize(z_result, p=2, dim=-1)
        return z_result


class DummySpatialResultEncoder(nn.Module):
    """Dummy Spatial Result Encoder."""
    def __init__(self, input_dim=768, output_dim=128):
        super().__init__()
        self.projector = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        z_result = self.projector(x)
        z_result = F.normalize(z_result, p=2, dim=-1)
        return z_result
