"""Video encoder using V-JEPA."""
from typing import Optional
import torch
import torch.nn as nn


class VideoEncoder(nn.Module):
    """Video encoder using V-JEPA (facebook/vjepa2-vitl-fpc64-256).
    
    Input: (batch, num_frames, channels, height, width)
    Output: (batch, seq_len, embed_dim)
    """
    
    def __init__(
        self,
        model_path: str = "models/vjepa2-vitl-fpc64-256",
        embed_dim: int = 1024,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize video encoder.
        
        Args:
            model_path: Path to pretrained V-JEPA model
            embed_dim: Output embedding dimension
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        try:
            from transformers import AutoModel
        except ImportError:
            raise RuntimeError("transformers is required for VideoEncoder")
        
        # Load pretrained V-JEPA model.
        # Prefer a local model directory if it exists. If local loading fails,
        # raise a clear error instead of silently falling back to online (which
        # can hang in offline environments).
        import os

        model_path_resolved = model_path
        if not os.path.isabs(model_path_resolved):
            model_path_resolved = os.path.join(os.getcwd(), model_path_resolved)

        # Require local model directory only. Fail fast with a helpful message
        # if the folder does not exist or loading fails.
        if os.path.exists(model_path_resolved):
            try:
                self.model = AutoModel.from_pretrained(
                    model_path_resolved,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load local V-JEPA model at '{model_path_resolved}': {e}.\n"
                    "Ensure the folder contains 'config.json' and model weights (pytorch_model.bin or model.safetensors),\n"
                    "and that required extras (e.g. 'safetensors') are installed."
                )
        else:
            raise FileNotFoundError(
                f"Local V-JEPA model directory not found: '{model_path_resolved}'.\n"
                "Please place the pretrained model in that path before running."
            )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Get model output dimension
        self.model_dim = self.model.config.hidden_size if hasattr(self.model.config, 'hidden_size') else 1024
        
        # Projection layer if needed
        if self.model_dim != embed_dim:
            self.projection = nn.Linear(self.model_dim, embed_dim)
        else:
            self.projection = nn.Identity()

        # Heuristic: inspect backbone for Conv3d vs Conv2d to decide expected
        # input layout. If a Conv3d is present, assume the backbone expects
        # video tensors (B, C, T, H, W). If the first conv is Conv2d, we
        # assume a per-frame image backbone (B, C, H, W) and will call it on
        # flattened frames. If unknown, self._expects_video remains None and
        # forward will raise a helpful error on mismatch.
        self._expects_video: Optional[bool] = None
        self._detected_conv_in_channels: Optional[int] = None
        try:
            for module in self.model.modules():
                if isinstance(module, nn.Conv3d):
                    self._expects_video = True
                    self._detected_conv_in_channels = getattr(module, 'in_channels', None)
                    break
                if isinstance(module, nn.Conv2d):
                    self._expects_video = False
                    self._detected_conv_in_channels = getattr(module, 'in_channels', None)
                    break
        except Exception:
            # Non-fatal: leave detection as None and provide diagnostic later
            self._expects_video = None
            self._detected_conv_in_channels = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Video tensor (batch, num_frames, channels, height, width)
        
        Returns:
            features: (batch, seq_len, embed_dim)
        """
        batch_size, num_frames, channels, height, width = x.shape

        # Try several reasonable ways to feed the pretrained model. Some V-JEPA
        # implementations expect a video tensor (B, C, T, H, W), others accept
        # (B, T, C, H, W) or only per-frame images. We try in order and fall
        # back to a lightweight dummy encoder if the pretrained model cannot be
        # used for the current input shape / local checkpoint.

        def _extract_features_from_outputs(outputs):
            if hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                return outputs[0]
            else:
                return outputs

        # Decide calling convention based on detection from __init__.
        if self._expects_video is True:
            # Many video transformers expect input in (B, T, C, H, W) and handle
            # internal permutation. Pass the tensor as-is to avoid double
            # permuting which can swap channels/frames.
            video_input = x
            if self._detected_conv_in_channels is not None and channels != self._detected_conv_in_channels:
                raise RuntimeError(
                    f"VideoEncoder: detected backbone first Conv3d expects {self._detected_conv_in_channels} channels, "
                    f"but input provides {channels}. Ensure frames are ordered as (B, T, C, H, W).")

            try:
                try:
                    outputs = self.model(video_input)
                except TypeError:
                    try:
                        outputs = self.model(video=video_input)
                    except TypeError:
                        outputs = self.model(pixel_values=video_input)

                features = _extract_features_from_outputs(outputs)
                features = self.projection(features)
                return features
            except Exception as e:
                raise RuntimeError(
                    f"VideoEncoder forward failed for video-style backbone: {e}. "
                    "Check that the local model matches expected V-JEPA layout and that inputs were preprocessed correctly.") from e

        elif self._expects_video is False:
            # Model expects per-frame images (Conv2d); flatten frames and call
            frames = x.reshape(batch_size * num_frames, channels, height, width)
            try:
                try:
                    outputs = self.model(frames)
                except TypeError:
                    outputs = self.model(pixel_values=frames)

                features = _extract_features_from_outputs(outputs)
                if features.dim() == 3:
                    seq_len = features.shape[1]
                    model_dim = features.shape[2]
                    features = features.reshape(batch_size, num_frames, seq_len, model_dim)
                    features = features.mean(dim=1)
                else:
                    features = features.reshape(batch_size, num_frames, -1).mean(dim=1)

                features = self.projection(features)
                return features
            except Exception as e:
                raise RuntimeError(
                    f"VideoEncoder forward failed for frame-wise backbone: {e}. "
                    "Check that the local model accepts per-frame images and that inputs were preprocessed correctly.") from e

        else:
            # Unknown detection: surface a clear error so user can inspect model
            raise RuntimeError(
                "VideoEncoder cannot determine backbone input type (video vs frame). "
                "Please verify the local pretrained model in 'models/' and ensure it contains a recognizable Conv2d/Conv3d layer, or avoid using pretrained model for tests.")


class DummyVideoEncoder(nn.Module):
    """Dummy video encoder for testing without pretrained model.
    
    Input: (batch, num_frames, channels, height, width)
    Output: (batch, seq_len, embed_dim)
    """
    
    def __init__(self, embed_dim: int = 1024, seq_len: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        
        # Simple conv + pool to reduce spatial dimensions
        self.conv = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((seq_len, 1, 1)),
        )
        self.proj = nn.Linear(128, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = self.conv(x)  # (B, 128, seq_len, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B, 128, seq_len)
        x = x.permute(0, 2, 1)  # (B, seq_len, 128)
        x = self.proj(x)  # (B, seq_len, embed_dim)
        return x


__all__ = ["VideoEncoder", "DummyVideoEncoder"]
