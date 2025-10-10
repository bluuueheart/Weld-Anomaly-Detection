"""Dataset configuration."""

# Data paths
DATA_ROOT = "Data"

# Video parameters  
VIDEO_NUM_FRAMES = 32
VIDEO_FRAME_SIZE = 224
VIDEO_FPS = None

# Post-weld image parameters
IMAGE_SIZE = 224
IMAGE_NUM_ANGLES = 5  # number of angles per sample

# Audio parameters
AUDIO_SAMPLE_RATE = 16000
AUDIO_N_MELS = 128
AUDIO_FRAMES = 256

# Sensor parameters
SENSOR_LEN = 256
SENSOR_CHANNELS = 6

# Category mapping
CATEGORIES = {
    "good_weld": 0,
    "crater_cracks": 1,
    "burn_through": 2,
    "excessive_penetration": 3,
    "porosity": 4,
    "spatter": 5,
}
