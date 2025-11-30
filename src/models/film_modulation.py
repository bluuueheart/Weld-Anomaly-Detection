"""
FiLM (Feature-wise Linear Modulation) module for Causal-FiLM.

This module implements sensor-guided modulation of high-dimensional features
(video and audio) using context extracted from low-dimensional sensor data.
"""

import torch
import torch.nn as nn
from typing import Tuple
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    """
    Mamba block with residual connection and layer normalization.
    """
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm is not installed. Please install it to use Mamba.")
            
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection: x + Mamba(Norm(x))
        # Standard Mamba architecture often uses Pre-Norm
        return x + self.mamba(self.norm(x))


class SensorModulator(nn.Module):
    """
    Sensor-guided FiLM modulation module.
    
    Extracts context from sensor time series using Mamba (State Space Model) and generates
    gamma (scale) and beta (shift) parameters for feature modulation.
    
    Architecture:
        sensor_data (B, T_sensor, D_sensor) 
        -> Linear (D_sensor -> hidden_dim)
        -> Mamba Blocks (num_layers)
        -> Last Token Pooling
        -> Linear(hidden_dim -> d_model * 2)
        -> split into gamma (B, 1, d_model) and beta (B, 1, d_model)
    
    Args:
        input_dim (int): Sensor input dimension (default: 6)
        hidden_dim (int): Mamba hidden dimension (default: 64)
        num_layers (int): Number of Mamba layers (default: 2)
        d_model (int): Target feature dimension for modulation (default: 128)
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        d_model: int = 128,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.d_model = d_model
        
        # Input projection
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Mamba layers for temporal context extraction
        self.layers = nn.Sequential(*[
            MambaBlock(
                dim=hidden_dim,
                d_state=16,
                d_conv=4,
                expand=2
            ) for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Linear layer to generate gamma and beta
        self.modulator_head = nn.Linear(hidden_dim, d_model * 2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        # Initialize modulator_head to output near-identity modulation
        nn.init.zeros_(self.modulator_head.weight)
        nn.init.zeros_(self.modulator_head.bias)
        
        # Initialize embedding
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)
    
    def forward(self, sensor_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to generate FiLM parameters.
        
        Args:
            sensor_data: Sensor time series (B, T_sensor, D_sensor)
            
        Returns:
            gamma: Scale parameter (B, 1, d_model)
            beta: Shift parameter (B, 1, d_model)
        """
        # Project input to hidden dimension
        x = self.embedding(sensor_data)  # (B, T, hidden_dim)
        
        # Pass through Mamba layers
        x = self.layers(x)  # (B, T, hidden_dim)
        x = self.norm(x)
        
        # Take the last token's hidden state which summarizes the sequence (causal)
        context_vector = x[:, -1, :]  # (B, hidden_dim)
        
        # Generate modulation parameters
        modulators = self.modulator_head(context_vector)  # (B, d_model * 2)
        
        # Split into gamma and beta
        gamma, beta = torch.chunk(modulators, 2, dim=1)  # Each: (B, d_model)
        
        # Add 1.0 to gamma for identity initialization (gamma=1, beta=0 initially)
        # This ensures modulation starts close to identity: F * (gamma+1) + beta ≈ F
        gamma = gamma + 1.0
        
        # Expand dimension for broadcasting: (B, 1, d_model)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        return gamma, beta


class DummySensorModulator(nn.Module):
    """
    Lightweight dummy sensor modulator for testing.
    
    Returns identity modulation (gamma=1, beta=0).
    """
    
    def __init__(self, d_model: int = 128, **kwargs):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, sensor_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return identity modulation.
        
        Args:
            sensor_data: Sensor time series (B, T_sensor, D_sensor)
            
        Returns:
            gamma: Ones (B, 1, d_model)
            beta: Zeros (B, 1, d_model)
        """
        batch_size = sensor_data.size(0)
        device = sensor_data.device
        
        gamma = torch.ones(batch_size, 1, self.d_model, device=device)
        beta = torch.zeros(batch_size, 1, self.d_model, device=device)
        
        return gamma, beta


def apply_film_modulation(
    features: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """
    Apply FiLM modulation to features.
    
    FiLM formula: F_modulated = F * gamma + beta
    
    Args:
        features: Input features (B, T, D)
        gamma: Scale parameter (B, 1, D)
        beta: Shift parameter (B, 1, D)
        
    Returns:
        modulated_features: FiLM-modulated features (B, T, D)
    """
    return features * gamma + beta
