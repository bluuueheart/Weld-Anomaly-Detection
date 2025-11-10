"""
Causal-FiLM Model for Unified Multimodal Anomaly Detection.

Implements the complete Causal-FiLM architecture:
- L0: Frozen feature extractors (V-JEPA, DINOv2, AST)
- L1: FiLM sensor modulation
- L2: Causal hierarchical encoders (Process + Result)
- L3: Anti-generalization decoder

Reference: README_v2.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .video_encoder import VideoEncoder, DummyVideoEncoder
from .image_encoder import ImageEncoder, DummyImageEncoder
from .audio_encoder import AudioEncoder, DummyAudioEncoder
from .film_modulation import SensorModulator, DummySensorModulator, apply_film_modulation
from .causal_encoders import ProcessEncoder, ResultEncoder, DummyProcessEncoder, DummyResultEncoder
from .causal_decoder import CausalDecoder, DummyCausalDecoder


class CausalFiLMModel(nn.Module):
    """
    Causal-FiLM: Causal-Hierarchical Fusion with Sensor-Modulation
    for Unified Multimodal Industrial Anomaly Detection.
    
    Architecture:
        L0 (Frozen Backbones):
            video -> V-JEPA -> F_video (B, T_v, 1024)
            audio -> AST -> F_audio (B, T_a, 768)
            images -> DINOv2 -> F_image (B, N, 768)
            sensor -> raw data (B, T_s, 6)
        
        Projection to unified D_model=128:
            F_video -> Linear -> (B, T_v, 128)
            F_audio -> Linear -> (B, T_a, 128)
            F_image -> Linear -> (B, N, 128)
        
        L1 (FiLM Modulation):
            sensor -> SensorModulator -> gamma, beta (B, 1, 128)
            F_video_mod = F_video * gamma + beta
            F_audio_mod = F_audio * gamma + beta
        
        L2 (Causal Encoders):
            Z_process = ProcessEncoder(F_video_mod, F_audio_mod)  # (B, 128)
            Z_result = ResultEncoder(F_image)  # (B, 128)
        
        L3 (Decoder):
            Z_result_pred = CausalDecoder(Z_process)  # (B, 128)
    
    Training:
        - Loss: L_recon + λ * L_text
        - Only "normal" samples used
    
    Inference:
        - Anomaly Score = 1 - cos_sim(Z_result, Z_result_pred)
    
    Args:
        video_config: Video encoder configuration
        image_config: Image encoder configuration
        audio_config: Audio encoder configuration
        d_model: Unified feature dimension (default: 128)
        sensor_input_dim: Sensor input dimension (default: 6)
        sensor_hidden_dim: SensorModulator hidden dim (default: 64)
        decoder_num_layers: Number of decoder layers (default: 2)
        decoder_dropout: Decoder dropout rate (default: 0.2)
        use_dummy: Use lightweight dummy encoders for testing
    """
    
    def __init__(
        self,
        video_config: Optional[Dict] = None,
        image_config: Optional[Dict] = None,
        audio_config: Optional[Dict] = None,
        d_model: int = 128,
        sensor_input_dim: int = 6,
        sensor_hidden_dim: int = 64,
        decoder_num_layers: int = 2,
        decoder_dropout: float = 0.2,
        use_dummy: bool = False,
    ):
        super().__init__()
        
        video_config = video_config or {}
        image_config = image_config or {}
        audio_config = audio_config or {}
        
        self.d_model = d_model
        self.use_dummy = use_dummy
        
        # ===== L0: Frozen Feature Extractors =====
        if use_dummy:
            self.video_encoder = DummyVideoEncoder(embed_dim=1024)
            self.audio_encoder = DummyAudioEncoder(embed_dim=768)
            self.image_encoder = DummyImageEncoder(
                embed_dim=768,
                num_angles=image_config.get("num_angles", 5),
            )
        else:
            # Video encoder (V-JEPA)
            self.video_encoder = VideoEncoder(
                model_path=video_config.get("model_name", "models/vjepa2-vitl-fpc64-256"),
                embed_dim=1024,
                freeze_backbone=True,  # Always frozen
            )
            
            # Audio encoder (AST)
            audio_model_path = audio_config.get("local_model_path") or \
                             audio_config.get("model_name", "MIT/ast-finetuned-audioset-14-14-0.443")
            self.audio_encoder = AudioEncoder(
                model_path=audio_model_path,
                embed_dim=768,
                freeze_backbone=True,  # Always frozen
            )
            
            # Image encoder (DINOv2)
            self.image_encoder = ImageEncoder(
                model_name=image_config.get("model_name", "facebook/dinov2-base"),
                embed_dim=768,
                num_angles=image_config.get("num_angles", 5),
                aggregation="none",  # We handle aggregation in ResultEncoder
                freeze_backbone=True,  # Always frozen
                local_model_path=image_config.get("local_model_path"),
            )
        
        # Projection layers to unified d_model=128
        self.video_projector = nn.Linear(1024, d_model)
        self.audio_projector = nn.Linear(768, d_model)
        self.image_projector = nn.Linear(768, d_model)
        
        # ===== L1: FiLM Sensor Modulation =====
        if use_dummy:
            self.sensor_modulator = DummySensorModulator(d_model=d_model)
        else:
            self.sensor_modulator = SensorModulator(
                input_dim=sensor_input_dim,
                hidden_dim=sensor_hidden_dim,
                num_layers=2,
                d_model=d_model,
            )
        
        # ===== L2: Causal Encoders =====
        if use_dummy:
            self.process_encoder = DummyProcessEncoder(d_model=d_model)
            self.result_encoder = DummyResultEncoder(d_model=d_model)
        else:
            self.process_encoder = ProcessEncoder(
                d_model=d_model,
                num_heads=4,
                num_layers=2,
                dropout=0.1,
            )
            self.result_encoder = ResultEncoder(
                d_model=d_model,
                dropout=0.1,
            )
        
        # ===== L3: Anti-Generalization Decoder =====
        if use_dummy:
            self.decoder = DummyCausalDecoder(
                d_model=d_model,
                dropout_p=decoder_dropout,
            )
        else:
            self.decoder = CausalDecoder(
                d_model=d_model,
                num_layers=decoder_num_layers,
                nhead=4,
                dropout_p=decoder_dropout,
            )
        
        # Output dimension
        self.output_dim = d_model
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_encodings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Causal-FiLM model.
        
        Args:
            batch: Dictionary containing:
                - 'video': (B, T, C, H, W) - in-process video
                - 'audio': (B, C, H, W) - mel spectrogram
                - 'post_weld_images': (B, N, C, H, W) - multi-angle images
                - 'sensor': (B, T_s, D_s) - sensor time series
            return_encodings: Return intermediate encodings (for analysis)
            
        Returns:
            Dictionary containing:
                - 'Z_result': Result encoding (B, d_model)
                - 'Z_result_pred': Reconstructed result encoding (B, d_model)
                - 'Z_process': Process encoding (B, d_model) [if return_encodings]
        """
        video = batch["video"]
        audio = batch["audio"]
        images = batch["post_weld_images"]
        sensor = batch["sensor"]
        
        # ===== L0: Extract Features from Frozen Backbones =====
        with torch.no_grad():
            F_video_raw = self.video_encoder(video)  # (B, T_v, 1024)
            F_audio_raw = self.audio_encoder(audio)  # (B, T_a, 768)
            F_image_raw = self.image_encoder(images)  # (B, N, 768)
        
        # Project to unified dimension d_model=128
        F_video = self.video_projector(F_video_raw)  # (B, T_v, 128)
        F_audio = self.audio_projector(F_audio_raw)  # (B, T_a, 128)
        F_image = self.image_projector(F_image_raw)  # (B, N, 128)
        
        # ===== L1: FiLM Sensor Modulation =====
        gamma, beta = self.sensor_modulator(sensor)  # (B, 1, 128)
        
        # Apply FiLM modulation to process modalities
        F_video_mod = apply_film_modulation(F_video, gamma, beta)  # (B, T_v, 128)
        F_audio_mod = apply_film_modulation(F_audio, gamma, beta)  # (B, T_a, 128)
        
        # ===== L2: Causal Hierarchical Encoding =====
        # Process encoder: video (modulated) attends to audio (modulated)
        Z_process = self.process_encoder(F_video_mod, F_audio_mod)  # (B, 128)
        
        # Result encoder: aggregate multi-view images
        Z_result = self.result_encoder(F_image)  # (B, 128)
        
        # ===== L3: Reconstruction via Decoder =====
        # During training, use noisy bottleneck; during inference, no noise
        is_training = self.training
        Z_result_pred = self.decoder(Z_process, is_training=is_training)  # (B, 128)
        
        # Build output
        output = {
            "Z_result": Z_result,
            "Z_result_pred": Z_result_pred,
        }
        
        if return_encodings:
            output["Z_process"] = Z_process
            output["gamma"] = gamma
            output["beta"] = beta
        
        return output
    
    def compute_anomaly_score(
        self,
        Z_result: torch.Tensor,
        Z_result_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute anomaly score based on reconstruction error.
        
        Args:
            Z_result: Ground truth result encoding (B, d_model)
            Z_result_pred: Reconstructed result encoding (B, d_model)
            
        Returns:
            anomaly_score: Anomaly score per sample (B,)
        """
        # Anomaly score = 1 - cosine_similarity
        # High score = poor reconstruction = anomaly
        cos_sim = F.cosine_similarity(Z_result, Z_result_pred, dim=1)
        anomaly_score = 1.0 - cos_sim
        return anomaly_score
    
    def get_feature_dim(self) -> int:
        """Get output feature dimension."""
        return self.output_dim
    
    def freeze_backbones(self):
        """Freeze all backbone encoders (V-JEPA, AST, DINOv2)."""
        for param in self.video_encoder.parameters():
            param.requires_grad = False
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """Get total number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


def create_causal_film_model(config: Dict, use_dummy: bool = False) -> CausalFiLMModel:
    """
    Factory function to create CausalFiLMModel from config.
    
    Args:
        config: Configuration dictionary with keys:
            - VIDEO_ENCODER
            - IMAGE_ENCODER
            - AUDIO_ENCODER
            - CAUSAL_FILM (optional, contains d_model, decoder_num_layers, etc.)
        use_dummy: Whether to use dummy encoders
        
    Returns:
        Initialized CausalFiLMModel
    """
    causal_film_config = config.get("CAUSAL_FILM", {})
    
    model = CausalFiLMModel(
        video_config=config.get("VIDEO_ENCODER", {}),
        image_config=config.get("IMAGE_ENCODER", {}),
        audio_config=config.get("AUDIO_ENCODER", {}),
        d_model=causal_film_config.get("d_model", 128),
        sensor_input_dim=causal_film_config.get("sensor_input_dim", 6),
        sensor_hidden_dim=causal_film_config.get("sensor_hidden_dim", 64),
        decoder_num_layers=causal_film_config.get("decoder_num_layers", 2),
        decoder_dropout=causal_film_config.get("decoder_dropout", 0.2),
        use_dummy=use_dummy,
    )
    
    return model
