"""
FiLM (Feature-wise Linear Modulation) module for Causal-FiLM.

This module implements sensor-guided modulation of high-dimensional features
(video and audio) using context extracted from low-dimensional sensor data.
"""

import torch
import torch.nn as nn
from typing import Tuple


class SensorModulator(nn.Module):
    """
    Sensor-guided FiLM modulation module.
    
    Extracts context from sensor time series using GRU and generates
    gamma (scale) and beta (shift) parameters for feature modulation.
    
    Architecture:
        sensor_data (B, T_sensor, D_sensor) 
        -> GRU (2 layers, hidden=64)
        -> Linear(hidden_dim -> d_model * 2)
        -> split into gamma (B, 1, d_model) and beta (B, 1, d_model)
    
    Args:
        input_dim (int): Sensor input dimension (default: 6)
        hidden_dim (int): GRU hidden dimension (default: 64)
        num_layers (int): Number of GRU layers (default: 2)
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
        
        # GRU for temporal context extraction
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        
        # Linear layer to generate gamma and beta
        self.modulator_head = nn.Linear(hidden_dim, d_model * 2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        # Initialize modulator_head to output near-identity modulation
        nn.init.zeros_(self.modulator_head.weight)
        nn.init.zeros_(self.modulator_head.bias)
    
    def forward(self, sensor_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to generate FiLM parameters.
        
        Args:
            sensor_data: Sensor time series (B, T_sensor, D_sensor)
            
        Returns:
            gamma: Scale parameter (B, 1, d_model)
            beta: Shift parameter (B, 1, d_model)
        """
        # Extract temporal context via GRU
        # We only care about the last hidden state which summarizes the sequence
        _, last_hidden_state = self.gru(sensor_data)  # (num_layers, B, hidden_dim)
        
        # Take the last layer's hidden state
        context_vector = last_hidden_state[-1]  # (B, hidden_dim)
        
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
