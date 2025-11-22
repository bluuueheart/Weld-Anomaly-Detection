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


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM).
    
    Learns to focus on salient features (anomalies) via a learnable power parameter p.
    As p -> infinity, GeM approaches Max Pooling.
    As p -> 1, GeM approaches Mean Pooling.
    """
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (B, C, N)
        # Pool over N (spatial dimension)
        return F.avg_pool1d(x.clamp(min=self.eps).pow(self.p), (x.size(-1))).pow(1./self.p).squeeze(-1)


class ResultEncoder(nn.Module):
    """
    Decoupled GeM Encoder (Dual-Stream).
    
    Splits processing into two streams:
    1. Texture Stream (Layer 8): Uses GeM Pooling to softly focus on anomalies.
    2. Structure Stream (Layer 11): Uses Mean Pooling to capture global structure.
    
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
        
        # GeM layer for Texture Stream
        self.gem_pool = GeM(p=3.0)
        
        # Stream 1: Texture (Layer 8)
        self.texture_projector = nn.Sequential(
            nn.Linear(768, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        
        # Stream 2: Structure (Layer 11)
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
                - 'mid': (B, N, SeqLen, 768) - Layer 8
                - 'high': (B, N, SeqLen, 768) - Layer 11
            
        Returns:
            (z_texture, z_structure): Tuple of (B, d_model) tensors
        """
        feat_mid = features_dict['mid']   # (B, N, SeqLen, 768)
        feat_high = features_dict['high'] # (B, N, SeqLen, 768)
        
        B, N, S, C = feat_mid.shape
        
        # Flatten views and sequence for global pooling context
        flat_mid = feat_mid.reshape(B, N * S, C)
        flat_high = feat_high.reshape(B, N * S, C)
        
        # --- Stream 1: Texture/Object (Layer 8) ---
        # Use GeM pooling
        # Permute to (B, C, N*S) for pooling over spatial dim
        x_mid = flat_mid.permute(0, 2, 1) # (B, 768, N*S)
        z_texture_raw = self.gem_pool(x_mid) # (B, 768)
        
        z_texture = self.texture_projector(z_texture_raw) # (B, d_model)
        z_texture = F.normalize(z_texture, p=2, dim=-1)

        # --- Stream 2: Structure (Layer 11) ---
        # Use Standard Mean Pooling
        z_structure_raw = flat_high.mean(dim=1) # (B, 768)
        
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
        # Dummy implementation
        feat = features_dict['mid'] # (B, N, S, 768)
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
