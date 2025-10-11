"""Audio encoder using AST (Audio Spectrogram Transformer)."""
from typing import Optional
import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """Audio encoder using AST (MIT/ast-finetuned-audioset-14-14-0.443).
    
    Input: (batch, 1, n_mels, time_frames) - mel spectrogram
    Output: (batch, seq_len, embed_dim)
    """
    
    def __init__(
        self,
        model_path: str = "models/ast-finetuned-audioset-14-14-0.443",
        embed_dim: int = 768,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize audio encoder.
        
        Args:
            model_path: Path to pretrained AST model
            embed_dim: Output embedding dimension
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        try:
            from transformers import AutoModel
        except ImportError:
            raise RuntimeError("transformers is required for AudioEncoder")
        
        # Load pretrained AST model from local folder only. Fail fast if missing.
        import os

        model_path_resolved = model_path
        if not os.path.isabs(model_path_resolved):
            model_path_resolved = os.path.join(os.getcwd(), model_path_resolved)

        if os.path.exists(model_path_resolved):
            try:
                self.model = AutoModel.from_pretrained(
                    model_path_resolved,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load local AST model at '{model_path_resolved}': {e}.\n"
                    "Ensure the folder contains 'config.json' and model weights (pytorch_model.bin or model.safetensors), and required packages are installed."
                )
        else:
            raise FileNotFoundError(
                f"Local AST model directory not found: '{model_path_resolved}'.\n"
                "Please place the pretrained model in that path before running."
            )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Get model output dimension
        self.model_dim = self.model.config.hidden_size if hasattr(self.model.config, 'hidden_size') else 768
        
        # Projection layer if needed
        if self.model_dim != embed_dim:
            self.projection = nn.Linear(self.model_dim, embed_dim)
        else:
            self.projection = nn.Identity()

        # Detect expected input shape from model config/architecture.
        # AST models typically expect specific mel-spectrogram dimensions.
        self._expected_input_shape: Optional[tuple] = None
        try:
            # Try to extract expected input shape from model config
            if hasattr(self.model.config, 'num_mel_bins'):
                expected_mels = self.model.config.num_mel_bins
            else:
                expected_mels = None
            
            if hasattr(self.model.config, 'max_length'):
                expected_time = self.model.config.max_length
            else:
                expected_time = None
            
            if expected_mels and expected_time:
                self._expected_input_shape = (expected_mels, expected_time)
        except Exception:
            self._expected_input_shape = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Mel spectrogram (batch, 1, n_mels, time_frames)
        
        Returns:
            features: (batch, seq_len, embed_dim)
        """
        # AST often expects (batch, n_mels, time_frames) or keyword 'inputs'
        def _extract_features_from_outputs(outputs):
            if hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                return outputs[0]
            else:
                return outputs

        # Normalize input to (B, n_mels, time_frames) for AST models
        if x.dim() == 4 and x.size(1) == 1:
            x_img = x.squeeze(1)  # (B, n_mels, time)
        elif x.dim() == 3:
            x_img = x
        else:
            raise RuntimeError(
                f"AudioEncoder expects a 3D or 4D tensor (got shape {tuple(x.shape)}). "
                "Ensure audio preprocessing produces (B, 1, n_mels, time_frames) or (B, n_mels, time_frames).")

        # Resize input to match model's expected shape if needed (via interpolation)
        if self._expected_input_shape is not None:
            expected_mels, expected_time = self._expected_input_shape
            current_mels, current_time = x_img.shape[-2], x_img.shape[-1]
            
            if (current_mels, current_time) != (expected_mels, expected_time):
                # Interpolate to match expected dimensions
                x_img = torch.nn.functional.interpolate(
                    x_img.unsqueeze(1),  # (B, 1, n_mels, time)
                    size=(expected_mels, expected_time),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)  # (B, n_mels, time)

        # Call model with properly shaped input
        try:
            try:
                outputs = self.model(x_img)
            except TypeError:
                try:
                    outputs = self.model(inputs=x_img)
                except TypeError:
                    outputs = self.model(pixel_values=x_img)

            features = _extract_features_from_outputs(outputs)
            features = self.projection(features)
            return features
        except Exception as e:
            # Provide a clear, non-silent failure so user can fix model weights
            # or preprocessing rather than silently using a dummy path.
            raise RuntimeError(f"AudioEncoder pretrained forward failed: {e}") from e


class DummyAudioEncoder(nn.Module):
    """Dummy audio encoder for testing without pretrained model.
    
    Input: (batch, 1, n_mels, time_frames)
    Output: (batch, seq_len, embed_dim)
    """
    
    def __init__(self, embed_dim: int = 768, seq_len: int = 32) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        
        # Simple conv layers
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((seq_len, 1)),
        )
        self.proj = nn.Linear(128, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.size(1) == 1:
            pass  # Already (B, 1, H, W)
        elif x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, H, W)
        
        x = self.conv(x)  # (B, 128, seq_len, 1)
        x = x.squeeze(-1)  # (B, 128, seq_len)
        x = x.permute(0, 2, 1)  # (B, seq_len, 128)
        x = self.proj(x)  # (B, seq_len, embed_dim)
        return x


__all__ = ["AudioEncoder", "DummyAudioEncoder"]
