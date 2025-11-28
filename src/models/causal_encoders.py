"""
Causal Encoders for Causal-FiLM.

Implements:
- ProcessEncoder: Encodes "process" modalities (video + audio) via cross-attention
- ResultEncoder: Encodes "result" modality (post-weld images) via MLP
"""

import torch
import torch.nn as nn


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
    Result Encoder for post-weld images.
    
    Architecture:
        - Multi-view image features (B, N_views, D) 
        - Mean pooling -> (B, D)
        - MLP (2-layer) -> (B, d_model)
    
    Args:
        d_model (int): Output feature dimension (default: 128)
        dropout (float): Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # MLP encoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
    
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Encode result modality (images).
        
        Args:
            image_features: Multi-view image features (B, N_views, d_model)
            
        Returns:
            Z_result: Result encoding (B, d_model)
        """
        # Mean pooling over views
        pooled_features = image_features.mean(dim=1)  # (B, d_model)
        
        # MLP encoding
        Z_result = self.encoder_mlp(pooled_features)  # (B, d_model)
        
        return Z_result


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
    
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Simple mean pooling."""
        return image_features.mean(dim=1)


class RobustResultEncoder(nn.Module):
    def __init__(self, output_dim=256): # Note: Updated output_dim to 256 to match your config
        super().__init__()
        
        # 1. Independent Norms (Keep this from Plan E)
        self.norm_l12 = nn.LayerNorm(768)
        self.norm_l8 = nn.LayerNorm(768)
        
        # 2. Adaptive Gating Network (The Upgrade)
        # Input: 1536 (768*2) -> Output: 1536 weights
        self.gate_net = nn.Sequential(
            nn.Linear(1536, 384), # Bottleneck for efficiency
            nn.ReLU(),
            nn.Linear(384, 1536),
            nn.Sigmoid()
        )
        
        # 3. Final Projector
        self.projector = nn.Sequential(
            nn.Linear(1536, 512),
            nn.LayerNorm(512),
            nn.SiLU(), # Swish/SiLU is better than ReLU for modern networks
            nn.Linear(512, output_dim)
        )

    def forward(self, dino_output):
        # Extract features (Same as Plan E)
        feat_l12 = dino_output['hidden_states'][11] 
        z_l12 = feat_l12.mean(dim=1) 
        feat_l8 = dino_output['hidden_states'][7]
        z_l8 = feat_l8.max(dim=1)[0] 

        # Normalize
        z_l12 = self.norm_l12(z_l12)
        z_l8 = self.norm_l8(z_l8)
        
        # Concat
        combined = torch.cat([z_l12, z_l8], dim=-1) # (B, 1536)
        
        # --- Adaptive Gating ---
        gates = self.gate_net(combined) # (B, 1536)
        gated_combined = combined * gates # Element-wise modulation
        
        # Project
        return self.projector(gated_combined)
