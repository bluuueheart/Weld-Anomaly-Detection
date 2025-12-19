"""
Quad-Modal SOTA Model for Welding Anomaly Detection.

Integrates four modality encoders with cross-attention fusion:
- Video: V-JEPA (in-process monitoring)
- Image: DINOv3 (post-weld inspection)
- Audio: AST (acoustic signals)
- Sensor: Transformer (multi-variate time series)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .video_encoder import VideoEncoder, DummyVideoEncoder
from .image_encoder import ImageEncoder, DummyImageEncoder
from .audio_encoder import AudioEncoder, DummyAudioEncoder
from .sensor_encoder import SensorEncoder, DummySensorEncoder
from .fusion import CrossAttentionFusionModule, DummyCrossAttentionFusion


class QuadModalSOTAModel(nn.Module):
    """
    Quad-modal SOTA model for welding anomaly detection.
    
    Architecture:
        Input: Batch dict with 'video', 'post_weld_images', 'audio', 'sensor'
        
        Step 1: Encode each modality
            - VideoEncoder:  (B, T, C, H, W) -> (B, V_seq, 1024)
            - ImageEncoder:  (B, N, C, H, W) -> (B, I_seq, 768)
            - AudioEncoder:  (B, C, H, W)    -> (B, A_seq, 768)
            - SensorEncoder: (B, T, D)       -> (B, S_seq, 256)
        
        Step 2: Cross-attention fusion
            - Fuse all modality features -> (B, hidden_dim)
        
        Output: Feature vector for downstream tasks (SupCon, k-NN)
    
    Args:
        video_config (dict): Video encoder configuration
        image_config (dict): Image encoder configuration
        audio_config (dict): Audio encoder configuration
        sensor_config (dict): Sensor encoder configuration
        fusion_config (dict): Fusion module configuration
        use_dummy (bool): Use lightweight dummy encoders for testing
    """
    
    def __init__(
        self,
        video_config: Optional[Dict] = None,
        image_config: Optional[Dict] = None,
        audio_config: Optional[Dict] = None,
        sensor_config: Optional[Dict] = None,
        fusion_config: Optional[Dict] = None,
        use_dummy: bool = False,
    ):
        super().__init__()
        
        # Default configurations
        video_config = video_config or {}
        image_config = image_config or {}
        audio_config = audio_config or {}
        sensor_config = sensor_config or {}
        fusion_config = fusion_config or {}
        
        self.use_dummy = use_dummy
        
        # Initialize encoders
        if use_dummy:
            self.video_encoder = DummyVideoEncoder(
                embed_dim=video_config.get("embed_dim", 1024)
            )
            self.image_encoder = DummyImageEncoder(
                embed_dim=image_config.get("embed_dim", 768),
                num_angles=image_config.get("num_angles", 5),
            )
            self.audio_encoder = DummyAudioEncoder(
                embed_dim=audio_config.get("embed_dim", 768)
            )
            self.sensor_encoder = DummySensorEncoder(
                input_dim=sensor_config.get("input_dim", 6),
                embed_dim=sensor_config.get("embed_dim", 256),
            )
            self.fusion = DummyCrossAttentionFusion(
                video_dim=video_config.get("embed_dim", 1024),
                image_dim=image_config.get("embed_dim", 768),
                audio_dim=audio_config.get("embed_dim", 768),
                sensor_dim=sensor_config.get("embed_dim", 256),
                hidden_dim=fusion_config.get("hidden_dim", 512),
            )
        else:
            self.video_encoder = VideoEncoder(
                model_path=video_config.get("model_name", "models/vjepa2-vitl-fpc64-256"),
                embed_dim=video_config.get("embed_dim", 1024),
                freeze_backbone=video_config.get("freeze_backbone", False),
            )
            self.image_encoder = ImageEncoder(
                model_name=image_config.get("model_name", "/root/work/models/dinov3-vith16plus-pretrain-lvd1689m"),
                embed_dim=image_config.get("embed_dim", 768),
                num_angles=image_config.get("num_angles", 5),
                aggregation=image_config.get("aggregation", "mean"),
                freeze_backbone=image_config.get("freeze_backbone", False),
                local_model_path=image_config.get("local_model_path"),
            )
            # AudioEncoder expects `model_path` for a local model directory.
            audio_model_path = audio_config.get("local_model_path") or audio_config.get("model_name", "MIT/ast-finetuned-audioset-14-14-0.443")
            self.audio_encoder = AudioEncoder(
                model_path=audio_model_path,
                embed_dim=audio_config.get("embed_dim", 768),
                freeze_backbone=audio_config.get("freeze_backbone", False),
            )
            self.sensor_encoder = SensorEncoder(
                input_dim=sensor_config.get("input_dim", 6),
                embed_dim=sensor_config.get("embed_dim", 256),
                num_heads=sensor_config.get("num_heads", 8),
                num_layers=sensor_config.get("num_layers", 4),
                dim_feedforward=sensor_config.get("dim_feedforward", 1024),
                dropout=sensor_config.get("dropout", 0.1),
            )
            self.fusion = CrossAttentionFusionModule(
                video_dim=video_config.get("embed_dim", 1024),
                image_dim=image_config.get("embed_dim", 768),
                audio_dim=audio_config.get("embed_dim", 768),
                sensor_dim=sensor_config.get("embed_dim", 256),
                hidden_dim=fusion_config.get("hidden_dim", 512),
                num_fusion_tokens=fusion_config.get("num_fusion_tokens", 4),
                num_heads=fusion_config.get("num_heads", 8),
                dropout=fusion_config.get("dropout", 0.1),
            )
        
        # Output dimension
        self.output_dim = fusion_config.get("hidden_dim", 512)
        
        # Feature normalization layer (helps with contrastive learning)
        self.feature_norm = nn.LayerNorm(self.output_dim)
        self.l2_normalize = fusion_config.get("l2_normalize", True)  # L2 normalization for contrastive learning
        
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through all encoders and fusion.
        
        Args:
            batch: Dictionary containing:
                - 'video': (B, T, C, H, W) - in-process video frames
                - 'post_weld_images': (B, N, C, H, W) - multi-angle post-weld images
                - 'audio': (B, C, H, W) - mel spectrogram
                - 'sensor': (B, T, D) - sensor time series
            return_attention: Whether to return attention weights
            
        Returns:
            fused_features: (B, hidden_dim) fused feature vector
            attention_weights (optional): Dict of attention weights if return_attention=True
        """
        # Extract modality data
        video = batch["video"]
        images = batch["post_weld_images"]
        audio = batch["audio"]
        sensor = batch["sensor"]
        
        # Encode each modality
        video_features = self.video_encoder(video)      # (B, V_seq, 1024)
        image_features = self.image_encoder(images)     # (B, I_seq, 768)
        audio_features = self.audio_encoder(audio)      # (B, A_seq, 768)
        sensor_features = self.sensor_encoder(sensor)   # (B, S_seq, 256)
        
        # Fuse all modalities
        fused_features = self.fusion(
            video_features=video_features,
            image_features=image_features,
            audio_features=audio_features,
            sensor_features=sensor_features,
            return_attention=return_attention,
        )
        
        # Apply feature normalization
        if not return_attention:
            # Only normalize features (not attention dict)
            fused_features = self.feature_norm(fused_features)
            
            # L2 normalization for contrastive learning
            if self.l2_normalize:
                fused_features = nn.functional.normalize(fused_features, p=2, dim=1)
        
        return fused_features
    
    def get_feature_dim(self) -> int:
        """Get output feature dimension."""
        return self.output_dim
    
    def freeze_encoders(self):
        """Freeze all encoder backbones (useful for fine-tuning fusion only)."""
        for param in self.video_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        for param in self.sensor_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoders(self):
        """Unfreeze all encoder backbones."""
        for param in self.video_encoder.parameters():
            param.requires_grad = True
        for param in self.image_encoder.parameters():
            param.requires_grad = True
        for param in self.audio_encoder.parameters():
            param.requires_grad = True
        for param in self.sensor_encoder.parameters():
            param.requires_grad = True
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """Get total number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


def create_quadmodal_model(config: Dict, use_dummy: bool = False) -> QuadModalSOTAModel:
    """
    Factory function to create QuadModalSOTAModel from config.
    
    Args:
        config: Configuration dictionary with keys:
            - VIDEO_ENCODER
            - IMAGE_ENCODER
            - AUDIO_ENCODER
            - SENSOR_ENCODER
            - FUSION
        use_dummy: Whether to use dummy encoders
        
    Returns:
        Initialized QuadModalSOTAModel
    """
    model = QuadModalSOTAModel(
        video_config=config.get("VIDEO_ENCODER", {}),
        image_config=config.get("IMAGE_ENCODER", {}),
        audio_config=config.get("AUDIO_ENCODER", {}),
        sensor_config=config.get("SENSOR_ENCODER", {}),
        fusion_config=config.get("FUSION", {}),
        use_dummy=use_dummy,
    )
    return model
