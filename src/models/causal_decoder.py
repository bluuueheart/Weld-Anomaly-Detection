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
    
    def __init__(self, d_model: int = 128, nhead: int = 4, dropout: float = 0.1):
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
    Dual-Stream Causal Decoder.
    
    Predicts both Texture and Structure vectors from the Process context.
    
    Args:
        d_model (int): Feature dimension
        num_layers (int): Number of transformer layers
        nhead (int): Number of attention heads
        dropout_p (float): Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 2,
        nhead: int = 4,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        
        # Shared Anti-Generalization Block
        # We use a deeper shared block to extract rich context
        self.shared_decoder = nn.Sequential(
            *[AntiGenBlock(d_model, nhead, dropout_p) for _ in range(num_layers)]
        )
        
        # Separate heads for Texture and Structure
        self.texture_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model) # Output Z_texture_pred
        )
        
        self.structure_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model) # Output Z_structure_pred
        )
        
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, z_process: torch.Tensor, is_training: bool = True) -> tuple:
        """
        Reconstruct result features from process features.
        
        Args:
            z_process: Process encoding (B, d_model)
            is_training: Whether in training mode (controls dropout)
            
        Returns:
            (z_texture_pred, z_structure_pred): Predicted vectors
        """
        # Add sequence dimension for transformer: (B, 1, D)
        x = z_process.unsqueeze(1)
        
        # Apply noisy bottleneck during training
        if is_training:
            x = self.dropout(x)
        
        # Shared decoding
        feat = self.shared_decoder(x) # (B, 1, D)
        feat = feat.squeeze(1) # (B, D)
        
        # Predict heads
        z_texture_pred = self.texture_head(feat)
        z_structure_pred = self.structure_head(feat)
        
        # Normalize outputs (since targets are normalized)
        z_texture_pred = F.normalize(z_texture_pred, p=2, dim=-1)
        z_structure_pred = F.normalize(z_structure_pred, p=2, dim=-1)
        
        return z_texture_pred, z_structure_pred


class DummyCausalDecoder(nn.Module):
    """Dummy decoder for testing."""
    
    def __init__(self, d_model: int = 128, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.tex_head = nn.Linear(d_model, d_model)
        self.struc_head = nn.Linear(d_model, d_model)
    
    def forward(self, z_process: torch.Tensor, **kwargs) -> tuple:
        z_tex = F.normalize(self.tex_head(z_process), p=2, dim=-1)
        z_struc = F.normalize(self.struc_head(z_process), p=2, dim=-1)
        return z_tex, z_struc


class SpatialCausalDecoder(nn.Module):
    """
    Spatial Causal Decoder.
    
    Generates spatial image features from global process vector using learnable positional embeddings.
    """
    def __init__(self, d_model=128, num_patches=256, num_layers=2, nhead=4, dropout_p=0.2):
        super().__init__()
        self.num_patches = num_patches
        self.d_model = d_model
        
        # 1. Learnable Positional Embeddings
        # Shape (1, N_patches, 128)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)
        
        # 2. Anti-Generalization Block
        self.layers = nn.ModuleList([AntiGenBlock(d_model, nhead) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, Z_process, is_training=True):
        # Z_process shape: (B, 128) -> Global features
        B = Z_process.shape[0]
        
        # 1. Noisy Bottleneck
        if is_training:
            Z_process = self.dropout(Z_process)
            
        # 2. Expand + Inject Position
        # (B, 128) -> (B, 1, 128) -> (B, N, 128)
        x = Z_process.unsqueeze(1).expand(-1, self.num_patches, -1)
        
        # Add positional info
        x = x + self.pos_embed
        
        # 3. Transformer layers
        for layer in self.layers:
            x = layer(x)
            
        # 4. Normalize output
        Z_result_pred = F.normalize(x, p=2, dim=-1)
        
        return Z_result_pred # (B, N_patches, 128)


class DummySpatialCausalDecoder(nn.Module):
    """Dummy Spatial Causal Decoder."""
    def __init__(self, d_model=128, num_patches=256):
        super().__init__()
        self.num_patches = num_patches
        self.d_model = d_model
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)
        
    def forward(self, Z_process, is_training=True):
        B = Z_process.shape[0]
        x = Z_process.unsqueeze(1).expand(-1, self.num_patches, -1)
        x = x + self.pos_embed
        Z_result_pred = F.normalize(x, p=2, dim=-1)
        return Z_result_pred
