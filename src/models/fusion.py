"""
Multimodal Cross-Attention Fusion Module.

This module implements the core fusion mechanism for the quad-modal SOTA model,
using learnable fusion tokens and cross-attention to dynamically integrate
features from four modalities: video, image, audio, and sensor.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


class CrossAttentionFusionModule(nn.Module):
    """
    Cross-attention fusion module for quad-modal integration.
    
    Architecture:
        1. Initialize learnable FUSION_TOKEN
        2. Cross-attend to each modality separately (video, image, audio, sensor)
        3. Aggregate attended features
        4. Project to unified output dimension
    
    Args:
        video_dim (int): Dimension of video features (default: 1024)
        image_dim (int): Dimension of image features (default: 768)
        audio_dim (int): Dimension of audio features (default: 768)
        sensor_dim (int): Dimension of sensor features (default: 256)
        hidden_dim (int): Hidden dimension for cross-attention (default: 512)
        num_fusion_tokens (int): Number of learnable fusion tokens (default: 4)
        num_heads (int): Number of attention heads (default: 8)
        dropout (float): Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        video_dim: int = 1024,
        image_dim: int = 768,
        audio_dim: int = 768,
        sensor_dim: int = 256,
        hidden_dim: int = 512,
        num_fusion_tokens: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.video_dim = video_dim
        self.image_dim = image_dim
        self.audio_dim = audio_dim
        self.sensor_dim = sensor_dim
        self.hidden_dim = hidden_dim
        self.num_fusion_tokens = num_fusion_tokens
        self.num_heads = num_heads
        
        # Learnable fusion tokens (query)
        self.fusion_tokens = nn.Parameter(
            torch.randn(1, num_fusion_tokens, hidden_dim)
        )
        nn.init.trunc_normal_(self.fusion_tokens, std=0.02)
        
        # Project each modality to hidden_dim (for key/value) with moderate dropout
        self.video_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(video_dim, hidden_dim),
        )
        self.image_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(image_dim, hidden_dim),
        )
        self.audio_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(audio_dim, hidden_dim),
        )
        self.sensor_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(sensor_dim, hidden_dim),
        )
        
        # Cross-attention layers for each modality
        self.video_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.image_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.audio_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.sensor_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer normalization for each modality
        self.video_norm = nn.LayerNorm(hidden_dim)
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.audio_norm = nn.LayerNorm(hidden_dim)
        self.sensor_norm = nn.LayerNorm(hidden_dim)
        
        # Aggregation: concat all attended features with moderate dropout
        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim * num_fusion_tokens * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Final projection layer with moderate dropout
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(
        self,
        video_features: torch.Tensor,
        image_features: torch.Tensor,
        audio_features: torch.Tensor,
        sensor_features: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of cross-attention fusion.
        
        Args:
            video_features: (batch_size, video_seq_len, video_dim)
            image_features: (batch_size, image_seq_len, image_dim)
            audio_features: (batch_size, audio_seq_len, audio_dim)
            sensor_features: (batch_size, sensor_seq_len, sensor_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            fused_features: (batch_size, hidden_dim)
            attention_weights (optional): Dict of attention weights for each modality
        """
        batch_size = video_features.size(0)
        
        # Project each modality to hidden_dim
        video_proj = self.video_proj(video_features)  # (B, V_seq, hidden)
        image_proj = self.image_proj(image_features)  # (B, I_seq, hidden)
        audio_proj = self.audio_proj(audio_features)  # (B, A_seq, hidden)
        sensor_proj = self.sensor_proj(sensor_features)  # (B, S_seq, hidden)
        
        # Expand fusion tokens for batch
        queries = self.fusion_tokens.expand(batch_size, -1, -1)  # (B, num_tokens, hidden)
        
        # Cross-attend to each modality
        video_attended, video_attn_weights = self.video_cross_attn(
            query=queries,
            key=video_proj,
            value=video_proj,
            need_weights=return_attention,
        )  # (B, num_tokens, hidden)
        video_attended = self.video_norm(video_attended + queries)
        
        image_attended, image_attn_weights = self.image_cross_attn(
            query=queries,
            key=image_proj,
            value=image_proj,
            need_weights=return_attention,
        )
        image_attended = self.image_norm(image_attended + queries)
        
        audio_attended, audio_attn_weights = self.audio_cross_attn(
            query=queries,
            key=audio_proj,
            value=audio_proj,
            need_weights=return_attention,
        )
        audio_attended = self.audio_norm(audio_attended + queries)
        
        sensor_attended, sensor_attn_weights = self.sensor_cross_attn(
            query=queries,
            key=sensor_proj,
            value=sensor_proj,
            need_weights=return_attention,
        )
        sensor_attended = self.sensor_norm(sensor_attended + queries)
        
        # Concatenate all attended features
        all_attended = torch.cat(
            [video_attended, image_attended, audio_attended, sensor_attended],
            dim=1,
        )  # (B, num_tokens * 4, hidden)
        
        # Flatten and aggregate
        all_attended_flat = all_attended.flatten(1)  # (B, num_tokens * 4 * hidden)
        aggregated = self.aggregation(all_attended_flat)  # (B, hidden)
        
        # Final projection
        fused_features = self.output_proj(aggregated)  # (B, hidden)
        
        if return_attention:
            attention_weights = {
                "video": video_attn_weights,
                "image": image_attn_weights,
                "audio": audio_attn_weights,
                "sensor": sensor_attn_weights,
            }
            return fused_features, attention_weights
        
        return fused_features


class DummyCrossAttentionFusion(nn.Module):
    """
    Lightweight dummy fusion module for testing without heavy dependencies.
    
    Simply concatenates mean-pooled features from all modalities.
    """
    
    def __init__(
        self,
        video_dim: int = 1024,
        image_dim: int = 768,
        audio_dim: int = 768,
        sensor_dim: int = 256,
        hidden_dim: int = 512,
        **kwargs,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Simple linear projections
        total_dim = video_dim + image_dim + audio_dim + sensor_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(total_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
    def forward(
        self,
        video_features: torch.Tensor,
        image_features: torch.Tensor,
        audio_features: torch.Tensor,
        sensor_features: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Simple concatenation-based fusion.
        
        Args:
            video_features: (batch_size, video_seq_len, video_dim)
            image_features: (batch_size, image_seq_len, image_dim)
            audio_features: (batch_size, audio_seq_len, audio_dim)
            sensor_features: (batch_size, sensor_seq_len, sensor_dim)
            
        Returns:
            fused_features: (batch_size, hidden_dim)
        """
        # Mean pooling over sequence dimension
        video_pooled = video_features.mean(dim=1)  # (B, video_dim)
        image_pooled = image_features.mean(dim=1)  # (B, image_dim)
        audio_pooled = audio_features.mean(dim=1)  # (B, audio_dim)
        sensor_pooled = sensor_features.mean(dim=1)  # (B, sensor_dim)
        
        # Concatenate all modalities
        concatenated = torch.cat(
            [video_pooled, image_pooled, audio_pooled, sensor_pooled],
            dim=1,
        )
        
        # Project to hidden_dim
        fused_features = self.fusion_proj(concatenated)
        
        if return_attention:
            # Return dummy attention weights
            batch_size = video_features.size(0)
            dummy_attn = {
                "video": torch.ones(batch_size, 1, 1),
                "image": torch.ones(batch_size, 1, 1),
                "audio": torch.ones(batch_size, 1, 1),
                "sensor": torch.ones(batch_size, 1, 1),
            }
            return fused_features, dummy_attn
        
        return fused_features
