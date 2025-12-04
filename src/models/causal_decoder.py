"""
Causal Decoder for Causal-FiLM.

Implements anti-generalization decoder with:
- Noisy Bottleneck (Dropout)
- Linear Attention (unfocused attention to prevent shortcut learning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    """
    Linear Attention module for unfocused attention.
    
    Unlike standard softmax attention which can focus sharply on specific tokens,
    linear attention provides a more diffuse attention pattern. This prevents
    the decoder from learning shortcuts and forces it to reconstruct from
    the global context.
    
    Implementation based on "Transformers are RNNs: Fast Autoregressive
    Transformers with Linear Attention" (Katharopoulos et al., 2020).
    
    Args:
        dim (int): Feature dimension
        heads (int): Number of attention heads
    """
    
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        
        assert dim % heads == 0, "dim must be divisible by heads"
        
        self.heads = heads
        self.dim_head = dim // heads
        
        # Linear projections for Q, K, V
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
        # Activation function for Q and K (ELU + 1 ensures positivity)
        self.activation = nn.ELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear attention.
        
        Args:
            x: Input features (B, SeqLen, Dim)
            
        Returns:
            out: Attended features (B, SeqLen, Dim)
        """
        B, N, C = x.shape
        
        # Project to Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 3 * (B, N, C)
        q, k, v = map(
            lambda t: t.view(B, N, self.heads, self.dim_head).transpose(1, 2),
            qkv
        )  # Each: (B, heads, N, dim_head)
        
        # Apply activation to Q and K (ensure positivity)
        q = self.activation(q) + 1.0  # (B, H, N, D)
        k = self.activation(k) + 1.0  # (B, H, N, D)
        
        # Linear attention mechanism
        # KV = K^T @ V: (B, H, D, D)
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        
        # Normalization factor: sum of K over sequence
        # Z = sum(K, dim=N): (B, H, D)
        z = k.sum(dim=2)  # (B, H, D)
        
        # Compute attention: Q @ KV / Z
        # (B, H, N, D) @ (B, H, D, E) -> (B, H, N, E)
        numerator = torch.einsum('bhnd,bhde->bhne', q, kv)
        # Normalize by Z (broadcast): (B, H, N, E) / (B, H, 1, E)
        denominator = torch.einsum('bhnd,bhd->bhn', q, z).unsqueeze(-1) + 1e-6
        out = numerator / denominator  # (B, H, N, D)
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.to_out(out)
        
        return out


class AntiGenBlock(nn.Module):
    """
    Anti-Generalization Transformer Block.
    
    Uses Linear Attention + Feed-Forward Network with residual connections.
    
    Args:
        d_model (int): Feature dimension
        nhead (int): Number of attention heads
        dropout (float): Dropout rate
    """
    
    def __init__(self, d_model: int = 256, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Linear attention (unfocused)
        self.attn = LinearAttention(d_model, nhead)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (B, SeqLen, d_model)
            
        Returns:
            out: Output features (B, SeqLen, d_model)
        """
        # Attention block with residual
        x = x + self.dropout(self.attn(self.norm1(x)))
        
        # FFN block with residual
        x = x + self.dropout(self.ffn(self.norm2(x)))
        
        return x


class CausalDecoder(nn.Module):
    """
    Causal Decoder with anti-generalization mechanisms.
    
    Key features:
    1. Noisy Bottleneck: Dropout on input to prevent overfitting
    2. Linear Attention: Unfocused attention to prevent shortcut learning
    3. Lightweight architecture: Only 2 layers to limit capacity
    
    Architecture:
        Z_process (B, d_model)
        -> Dropout (Noisy Bottleneck)
        -> Unsqueeze to (B, 1, d_model)
        -> N x AntiGenBlock (Linear Attention + FFN)
        -> Squeeze to (B, d_model)
        -> Z_result_pred
    
    Args:
        d_model (int): Feature dimension (default: 256)
        num_layers (int): Number of decoder layers (default: 2)
        nhead (int): Number of attention heads (default: 4)
        dropout_p (float): Dropout probability for noisy bottleneck (default: 0.2)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 2,
        nhead: int = 4,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        
        # Noisy bottleneck (applied during training only)
        self.noisy_dropout = nn.Dropout(dropout_p)
        
        # Anti-generalization blocks
        self.layers = nn.ModuleList([
            AntiGenBlock(d_model, nhead, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Output normalization
        self.output_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        Z_process: torch.Tensor,
        is_training: bool = True,
    ) -> torch.Tensor:
        """
        Decode process encoding to reconstruct result encoding.
        
        Args:
            Z_process: Process encoding (B, d_model)
            is_training: Whether in training mode (applies noisy bottleneck)
            
        Returns:
            Z_result_pred: Reconstructed result encoding (B, d_model)
        """
        # 1. Apply noisy bottleneck (only during training)
        if is_training:
            Z_process_noisy = self.noisy_dropout(Z_process)
        else:
            Z_process_noisy = Z_process
        
        # 2. Reshape for transformer: (B, d_model) -> (B, 1, d_model)
        x = Z_process_noisy.unsqueeze(1)  # (B, 1, d_model)
        
        # 3. Pass through anti-generalization blocks
        for layer in self.layers:
            x = layer(x)  # (B, 1, d_model)
        
        # 4. Reshape back: (B, 1, d_model) -> (B, d_model)
        Z_result_pred = x.squeeze(1)  # (B, d_model)
        
        # 5. Output normalization
        Z_result_pred = self.output_norm(Z_result_pred)
        
        return Z_result_pred


class DummyCausalDecoder(nn.Module):
    """
    Lightweight dummy decoder for testing.
    
    Simply applies a linear layer with dropout.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        dropout_p: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout_p)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
    
    def forward(
        self,
        Z_process: torch.Tensor,
        is_training: bool = True,
    ) -> torch.Tensor:
        """Simple linear decoding."""
        if is_training:
            Z_process = self.dropout(Z_process)
        return self.decoder(Z_process)


class AntiGenDecoder(nn.Module):
    """
    Simple, single-stream AntiGenDecoder.
    Process -> 512. No dual heads.
    """
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(0.1),  # Added Dropout (Noisy Bottleneck effect)
            nn.Linear(d_model, d_model)
            # No LayerNorm or activation at the end to allow values to grow freely
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
