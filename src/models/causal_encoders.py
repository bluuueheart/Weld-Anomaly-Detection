"""
Causal Encoders for Causal-FiLM.

Implements:
- ProcessEncoder: Encodes "process" modalities (video + audio) via cross-attention
- ResultEncoder: Encodes "result" modality (post-weld images) via MLP
- GeM Pooling: Generalized Mean Pooling with learnable parameter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPooling(nn.Module):
    """
    Generalized Mean (GeM) Pooling.
    
    GeM pooling generalizes average pooling and max pooling by introducing
    a learnable parameter p that controls the pooling behavior:
    - p=1: equivalent to average pooling
    - p→∞: approaches max pooling
    - p>1: emphasizes larger activations
    
    Formula: (1/N * Σ(x_i^p))^(1/p)
    
    Args:
        p (float): Initial value of the learnable parameter (default: 6.0)
        eps (float): Small value to avoid numerical issues (default: 1e-6)
    """
    
    def __init__(self, p=6.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)  # Learnable parameter
        self.eps = eps
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, N, D) where N is sequence length
            
        Returns:
            Pooled tensor of shape (B, D)
        """
        # Clamp p to avoid numerical instability
        p = self.p.clamp(min=1.0)
        
        # GeM pooling: (mean(x^p))^(1/p)
        return (x.clamp(min=self.eps).pow(p).mean(dim=1)).pow(1.0 / p)


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
        d_model (int): Feature dimension (default: 256)
        num_heads (int): Number of attention heads (default: 4)
        num_layers (int): Number of cross-attention layers (default: 2)
        dropout (float): Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int = 256,
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
        d_model (int): Output feature dimension (default: 256)
        dropout (float): Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int = 256,
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
    
    def __init__(self, d_model: int = 256, **kwargs):
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
    
    def __init__(self, d_model: int = 256, **kwargs):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Simple mean pooling."""
        return image_features.mean(dim=1)


class RobustResultEncoder(nn.Module):
    def __init__(self, output_dim=512, input_dim=1280): # Updated for DINOv3-vit-h/16+ (1280-dim)
        super().__init__()
        
        # 1. Independent Norms (Updated for DINOv3's 1280-dim output)
        # Layer 21: Use both Mean and GeM pooling for texture details
        self.norm_l21_mean = nn.LayerNorm(input_dim)  # Layer 21 Mean: general texture
        self.norm_l21_gem = nn.LayerNorm(input_dim)   # Layer 21 GeM: learnable-weighted texture features
        self.norm_l32 = nn.LayerNorm(input_dim)       # Layer 32: deepest semantic features
        
        # GeM Pooling for Layer 21 (learnable parameter p)
        self.gem_pooling = GeMPooling(p=6.0)  # Initialize p=6.0 (balance between stability and max-like behavior)
        
        # 2. Adaptive Gating Network (The Upgrade)
        # Input: input_dim*3 (1280*3 = 3840) -> Output: 3840 weights
        # Now includes L21_mean + L21_gem + L32
        concat_dim = input_dim * 3
        self.gate_net = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 4),  # Bottleneck for efficiency
            nn.SiLU(),
            nn.Linear(concat_dim // 4, concat_dim),
            nn.Sigmoid()
        )
        
        # 3. Final Projector
        self.projector = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(), # Swish/SiLU is better than ReLU for modern networks
            # Removed Dropout here to prevent underfitting
            nn.Linear(512, output_dim)
            # No LayerNorm or activation at the end to allow values to grow freely
        )

    def forward(self, dino_output):
        # Extract features from Layer 21 (mid-level) and Layer 32 (deepest)
        feat_l21 = dino_output['hidden_states'][20]  # Layer 21 (index 20)
        
        # Layer 21: Use BOTH Mean and GeM pooling to capture different texture aspects
        z_l21_mean = feat_l21.mean(dim=1)  # Mean pool: general texture patterns
        z_l21_gem = self.gem_pooling(feat_l21)  # GeM pool: learnable-weighted texture features
        
        # Layer 32: Use Mean pooling for deepest semantic features
        feat_l32 = dino_output['hidden_states'][31]  # Layer 32 (index 31)
        z_l32 = feat_l32.mean(dim=1)  # Mean pool for deepest semantic features

        # Normalize each component independently
        z_l21_mean = self.norm_l21_mean(z_l21_mean)
        z_l21_gem = self.norm_l21_gem(z_l21_gem)
        z_l32 = self.norm_l32(z_l32)
        
        # Concat: L21_mean + L21_gem + L32
        combined = torch.cat([z_l21_mean, z_l21_gem, z_l32], dim=-1) # (B, 3840)
        
        # --- Adaptive Gating ---
        gates = self.gate_net(combined) # (B, 3840)
        gated_combined = combined * gates # Element-wise modulation
        
        # Project
        return self.projector(gated_combined)
