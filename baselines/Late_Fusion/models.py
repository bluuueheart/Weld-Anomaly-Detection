"""Auto-encoder models for Late Fusion baseline.

Implements the audio and video auto-encoders described in the paper:
- Audio: 1D CNN auto-encoder for spectrogram reconstruction
- Video: Two-stage model with frozen SlowFast + trainable FC auto-encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AudioAutoEncoder(nn.Module):
    """1D CNN Auto-Encoder for audio anomaly detection.
    
    Architecture from Table 5 in paper:
    - Encoder: BatchNorm -> Conv(n_bins->1024) -> 3xConv(1024->1024) -> Conv(1024->bottleneck)
    - Decoder: ConvTranspose(bottleneck->1024) -> 3xConvTranspose(1024->1024) -> ConvTranspose(1024->n_bins)
    - Activation: Leaky-ReLU for all layers except last (PReLU)
    
    Input: (batch, n_bins, time_steps) - STFT spectrogram
    Output: (batch, n_bins, time_steps) - Reconstructed spectrogram
    """
    
    def __init__(
        self,
        n_bins: int = 8193,  # n_fft // 2 + 1 for n_fft=16384
        bottleneck_dim: int = 48,
        hidden_channels: int = 1024,
        num_conv_layers: int = 3,
    ):
        """Initialize audio auto-encoder.
        
        Args:
            n_bins: Number of frequency bins in spectrogram
            bottleneck_dim: Dimension of bottleneck layer
            hidden_channels: Number of channels in hidden conv layers
            num_conv_layers: Number of 1024->1024 conv layers
        """
        super().__init__()
        self.n_bins = n_bins
        self.bottleneck_dim = bottleneck_dim
        
        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.BatchNorm1d(n_bins))
        encoder_layers.append(nn.Conv1d(n_bins, hidden_channels, kernel_size=3, stride=1, padding=1))
        encoder_layers.append(nn.LeakyReLU(0.2))
        
        # Hidden conv layers
        for _ in range(num_conv_layers):
            encoder_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1))
            encoder_layers.append(nn.LeakyReLU(0.2))
        
        # Bottleneck
        encoder_layers.append(nn.Conv1d(hidden_channels, bottleneck_dim, kernel_size=3, stride=1, padding=1))
        encoder_layers.append(nn.LeakyReLU(0.2))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.ConvTranspose1d(bottleneck_dim, hidden_channels, kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.LeakyReLU(0.2))
        
        # Hidden deconv layers
        for _ in range(num_conv_layers):
            decoder_layers.append(nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1))
            decoder_layers.append(nn.LeakyReLU(0.2))
        
        # Output layer with PReLU
        decoder_layers.append(nn.ConvTranspose1d(hidden_channels, n_bins, kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.PReLU())
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input spectrogram (batch, n_bins, time_steps)
            
        Returns:
            Reconstructed spectrogram (batch, n_bins, time_steps)
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute frame-wise reconstruction error (anomaly score).
        
        Args:
            x: Input spectrogram (batch, n_bins, time_steps)
            
        Returns:
            Frame-wise MSE (batch, time_steps)
        """
        reconstruction = self.forward(x)
        # MSE per frame
        error = F.mse_loss(reconstruction, x, reduction='none')  # (batch, n_bins, time_steps)
        error = error.mean(dim=1)  # Average over frequency bins -> (batch, time_steps)
        return error


class VideoAutoEncoder(nn.Module):
    """Two-stage video anomaly detection model.
    
    Stage 1: Frozen SlowFast for feature extraction
    Stage 2: FC Auto-encoder for anomaly detection
    
    Architecture from Table 6 in paper:
    - Encoder: Linear(2304->512) -> Linear(512->256) -> Dropout(0.5) -> 
               Linear(256->128) -> Dropout(0.5) -> Linear(128->64) -> Dropout(0.5) -> Linear(64->64)
    - Decoder: Linear(64->64) -> Linear(64->128) -> Linear(128->256) -> 
               Linear(256->512) -> Dropout(0.5) -> Linear(512->2304)
    - Activation: ReLU after all linear layers
    
    Input: (batch, num_frames, channels, height, width) - Video frames
    Output: (batch, num_frames, feature_dim) - Reconstructed features
    """
    
    def __init__(
        self,
        feature_dim: int = 2304,
        encoder_layers: list = None,
        decoder_layers: list = None,
        dropout: float = 0.5,
        slowfast_model: Optional[nn.Module] = None,
    ):
        """Initialize video auto-encoder.
        
        Args:
            feature_dim: Dimension of SlowFast features
            encoder_layers: List of layer dimensions for encoder
            decoder_layers: List of layer dimensions for decoder
            dropout: Dropout probability
            slowfast_model: Pre-initialized frozen SlowFast model (optional)
        """
        super().__init__()
        
        if encoder_layers is None:
            encoder_layers = [2304, 512, 256, 128, 64, 64]
        if decoder_layers is None:
            decoder_layers = [64, 64, 128, 256, 512, 2304]
            
        self.feature_dim = feature_dim
        self.dropout = dropout
        
        # Stage 1: Frozen SlowFast (if provided)
        self.slowfast = slowfast_model
        if self.slowfast is not None:
            for param in self.slowfast.parameters():
                param.requires_grad = False
            self.slowfast.eval()
        
        # Stage 2: FC Auto-encoder
        # Encoder
        encoder_modules = []
        for i in range(len(encoder_layers) - 1):
            encoder_modules.append(nn.Linear(encoder_layers[i], encoder_layers[i+1]))
            encoder_modules.append(nn.ReLU())
            # Add dropout after specific layers (after 256, 128, 64 as per paper)
            if i >= 1 and i < len(encoder_layers) - 2:
                encoder_modules.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*encoder_modules)
        
        # Decoder
        decoder_modules = []
        for i in range(len(decoder_layers) - 1):
            decoder_modules.append(nn.Linear(decoder_layers[i], decoder_layers[i+1]))
            if i < len(decoder_layers) - 2:  # ReLU except for last layer
                decoder_modules.append(nn.ReLU())
            # Add dropout before final layer (after 512 as per paper)
            if i == len(decoder_layers) - 2:
                decoder_modules.insert(-1, nn.Dropout(dropout))
        
        self.decoder = nn.Sequential(*decoder_modules)
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using SlowFast (Stage 1).
        
        Args:
            x: Video frames (batch, num_frames, channels, height, width)
            
        Returns:
            Features (batch, num_frames, feature_dim)
        """
        if self.slowfast is None:
            # If no SlowFast model, return dummy features for testing
            batch_size, num_frames = x.shape[0], x.shape[1]
            return torch.zeros(batch_size, num_frames, self.feature_dim, device=x.device)
        
        batch_size, num_frames, C, H, W = x.shape
        
        # Process each frame independently or use sliding window
        # For simplicity, we'll process frames independently
        # In practice, SlowFast expects clip-level input
        features = []
        
        with torch.no_grad():
            for i in range(num_frames):
                # Extract single frame feature
                frame = x[:, i]  # (batch, C, H, W)
                feat = self.slowfast(frame)  # (batch, feature_dim)
                features.append(feat)
        
        features = torch.stack(features, dim=1)  # (batch, num_frames, feature_dim)
        return features
        
    def forward(self, x: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Video frames (batch, num_frames, C, H, W) or pre-extracted features
            features: Pre-extracted features (batch, num_frames, feature_dim), if provided
            
        Returns:
            Reconstructed features (batch, num_frames, feature_dim)
        """
        # Extract features if not provided
        if features is None:
            if x.dim() == 5:  # Video input
                features = self.extract_features(x)
            else:  # Already features
                features = x
        
        # Encode and decode
        batch_size, num_frames, feat_dim = features.shape
        features_flat = features.reshape(-1, feat_dim)  # (batch*num_frames, feature_dim)
        
        latent = self.encoder(features_flat)
        reconstruction = self.decoder(latent)
        
        reconstruction = reconstruction.reshape(batch_size, num_frames, feat_dim)
        return reconstruction
    
    def get_reconstruction_error(self, x: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute frame-wise reconstruction error (anomaly score).
        
        Args:
            x: Video frames or features
            features: Pre-extracted features (optional)
            
        Returns:
            Frame-wise MSE (batch, num_frames)
        """
        if features is None:
            if x.dim() == 5:
                features = self.extract_features(x)
            else:
                features = x
                
        reconstruction = self.forward(x, features)
        
        # MSE per frame
        error = F.mse_loss(reconstruction, features, reduction='none')  # (batch, num_frames, feature_dim)
        error = error.mean(dim=2)  # Average over feature dimension -> (batch, num_frames)
        return error


class LateFusionModel(nn.Module):
    """Late Fusion model combining audio and video auto-encoders.
    
    This model:
    1. Computes anomaly scores from audio and video separately
    2. Standardizes scores using training set statistics
    3. Combines scores using weighted sum
    """
    
    def __init__(
        self,
        audio_model: AudioAutoEncoder,
        video_model: VideoAutoEncoder,
        audio_weight: float = 0.37,
        video_weight: float = 0.63,
    ):
        """Initialize late fusion model.
        
        Args:
            audio_model: Trained audio auto-encoder
            video_model: Trained video auto-encoder
            audio_weight: Weight for audio scores (w in paper)
            video_weight: Weight for video scores (1-w in paper)
        """
        super().__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.audio_weight = audio_weight
        self.video_weight = video_weight
        
        # Statistics for standardization (computed from training set)
        self.register_buffer('audio_mean', torch.tensor(0.0))
        self.register_buffer('audio_std', torch.tensor(1.0))
        self.register_buffer('video_mean', torch.tensor(0.0))
        self.register_buffer('video_std', torch.tensor(1.0))
        
    def set_standardization_stats(
        self,
        audio_mean: float,
        audio_std: float,
        video_mean: float,
        video_std: float,
    ):
        """Set standardization statistics from training set.
        
        Args:
            audio_mean: Mean of audio anomaly scores on training set
            audio_std: Std of audio anomaly scores on training set
            video_mean: Mean of video anomaly scores on training set
            video_std: Std of video anomaly scores on training set
        """
        self.audio_mean = torch.tensor(audio_mean)
        self.audio_std = torch.tensor(audio_std)
        self.video_mean = torch.tensor(video_mean)
        self.video_std = torch.tensor(video_std)
        
    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        video_features: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Compute fused anomaly scores.
        
        Args:
            audio: Audio spectrograms (batch, n_bins, time_steps)
            video: Video frames or features
            video_features: Pre-extracted video features (optional)
            
        Returns:
            Tuple of (fused_scores, audio_scores, video_scores)
            - fused_scores: (batch, max_time_steps)
            - audio_scores: (batch, audio_time_steps)
            - video_scores: (batch, video_time_steps)
        """
        # Get reconstruction errors
        audio_scores = self.audio_model.get_reconstruction_error(audio)
        video_scores = self.video_model.get_reconstruction_error(video, video_features)
        
        # Standardize scores
        audio_scores_std = (audio_scores - self.audio_mean) / (self.audio_std + 1e-8)
        video_scores_std = (video_scores - self.video_mean) / (self.video_std + 1e-8)
        
        # Align time dimensions if needed (simple repeat for now)
        batch_size = audio_scores_std.shape[0]
        max_time = max(audio_scores_std.shape[1], video_scores_std.shape[1])
        
        if audio_scores_std.shape[1] < max_time:
            audio_scores_std = F.interpolate(
                audio_scores_std.unsqueeze(1),
                size=max_time,
                mode='linear',
                align_corners=False
            ).squeeze(1)
        
        if video_scores_std.shape[1] < max_time:
            video_scores_std = F.interpolate(
                video_scores_std.unsqueeze(1),
                size=max_time,
                mode='linear',
                align_corners=False
            ).squeeze(1)
        
        # Weighted fusion
        fused_scores = self.audio_weight * audio_scores_std + self.video_weight * video_scores_std
        
        return fused_scores, audio_scores, video_scores
