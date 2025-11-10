"""Models module."""

from .video_encoder import VideoEncoder
from .audio_encoder import AudioEncoder
from .sensor_encoder import SensorEncoder
from .image_encoder import ImageEncoder
from .fusion import CrossAttentionFusionModule, DummyCrossAttentionFusion
from .quadmodal_model import QuadModalSOTAModel, create_quadmodal_model

# Causal-FiLM model components
from .film_modulation import SensorModulator, DummySensorModulator, apply_film_modulation
from .causal_encoders import ProcessEncoder, ResultEncoder, DummyProcessEncoder, DummyResultEncoder
from .causal_decoder import CausalDecoder, DummyCausalDecoder
from .causal_film_model import CausalFiLMModel, create_causal_film_model

__all__ = [
    "VideoEncoder",
    "AudioEncoder",
    "SensorEncoder",
    "ImageEncoder",
    "CrossAttentionFusionModule",
    "DummyCrossAttentionFusion",
    "QuadModalSOTAModel",
    "create_quadmodal_model",
    # Causal-FiLM
    "SensorModulator",
    "DummySensorModulator",
    "apply_film_modulation",
    "ProcessEncoder",
    "ResultEncoder",
    "DummyProcessEncoder",
    "DummyResultEncoder",
    "CausalDecoder",
    "DummyCausalDecoder",
    "CausalFiLMModel",
    "create_causal_film_model",
]
