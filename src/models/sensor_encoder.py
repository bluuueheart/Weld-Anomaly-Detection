"""Sensor encoder using Transformer."""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch, embed_dim)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SensorEncoder(nn.Module):
    """Sensor encoder using Transformer for time series data.
    
    Input: (batch, seq_len, input_dim) - sensor time series
    Output: (batch, seq_len, embed_dim)
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        """Initialize sensor encoder.
        
        Args:
            input_dim: Input feature dimension (number of sensor channels)
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # Will use (seq, batch, embed)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Sensor data (batch, seq_len, input_dim)
            mask: Attention mask (optional)
        
        Returns:
            features: (batch, seq_len, embed_dim)
        """
        # Project input to embedding dimension
        x = self.input_proj(x)  # (batch, seq_len, embed_dim)
        
        # Transpose for transformer: (seq_len, batch, embed_dim)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer
        x = self.transformer(x, mask=mask)
        
        # Transpose back: (batch, seq_len, embed_dim)
        x = x.transpose(0, 1)
        
        # Layer norm
        x = self.norm(x)
        
        return x


class DummySensorEncoder(nn.Module):
    """Dummy sensor encoder for simple baseline.
    
    Input: (batch, seq_len, input_dim)
    Output: (batch, seq_len, embed_dim)
    """
    
    def __init__(self, input_dim: int = 6, embed_dim: int = 256) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Simple MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.mlp(x)
        x = self.norm(x)
        return x


__all__ = ["SensorEncoder", "DummySensorEncoder", "PositionalEncoding"]
