"""Dataset configuration.

This module contains dataset-level constants and backward-compatible aliases
for older code and tests.
"""

# Data paths
# Full dataset path on AutoDL server
DATA_ROOT = "/root/autodl-tmp/Intel_Robotic_Welding_Multimodal_Dataset/raid/intel_robotic_welding_dataset"

# Manifest file for train/test split
MANIFEST_PATH = "configs/manifest.csv"

# Video parameters
VIDEO_NUM_FRAMES = 32
VIDEO_FRAME_SIZE = 224
VIDEO_FPS = None

# Backward-compatibility alias: some tests import VIDEO_LENGTH
VIDEO_LENGTH = VIDEO_NUM_FRAMES

# Post-weld image parameters
IMAGE_SIZE = 224
IMAGE_NUM_ANGLES = 5  # number of angles per sample

# Audio parameters
AUDIO_SAMPLE_RATE = 16000
AUDIO_N_MELS = 128
AUDIO_FRAMES = 256

# Backward-compatible audio alias: older code may expect AUDIO_DURATION
AUDIO_DURATION = AUDIO_FRAMES

# Sensor parameters
SENSOR_LEN = 256
SENSOR_CHANNELS = 6

# Backward-compatible sensor alias: tests may expect SENSOR_LENGTH
SENSOR_LENGTH = SENSOR_LEN

# Category mapping
# Categories expected in folder names. These keys are matched (case-insensitive)
# against folder names. The matching logic in `src.dataset` prefers the longest
# matching key when multiple keys appear in a folder name (so e.g. the
# "porosity_w-excessive_penetration" key will be chosen over "porosity" when
# both substrings are present).
#
# The integer values are stable labels returned by the dataset.
CATEGORIES = {
    "good": 0,
    "excessive_convexity": 1,
    "undercut": 2,
    "lack_of_fusion": 3,
    "porosity_w-excessive_penetration": 4,
    "porosity": 5,
    "spatter": 6,
    "burnthrough": 7,
    "excessive_penetration": 8,
    "crater_cracks": 9,
    "warping": 10,
    "overlap": 11,
}

# Augmentation configuration: probabilities and parameter ranges used by
# src.dataset's augmentation helpers. Kept lightweight and optional so code
# continues to work when import fails (defaults are provided in code).
AUGMENTATION = {
    # spatial
    "p_hflip": 0.5,
    "p_vflip": 0.0,
    "p_rotate": 0.3,
    "rotate_max_deg": 10.0,
    "p_random_resized_crop": 0.3,
    "rrc_scale_min": 0.8,

    # color
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.1,
    "hue": 0.05,

    # blur / noise
    "p_blur": 0.2,
    "blur_max_ksize": 5,
    "p_gauss_noise": 0.2,
    "gauss_noise_sigma": 0.02,

    # erasing / cutout
    "p_cutout": 0.25,
    "cutout_area_min": 0.02,
    "cutout_area_max": 0.2,

    # video-specific
    "p_temporal_shift": 0.2,
    "max_temporal_shift": 2,

    # audio (specaugment)
    "p_freq_mask": 0.5,
    "freq_mask_max_band": 0.2,  # fraction of n_mels
    "p_time_mask": 0.5,
    "time_mask_max_band": 0.15,

    # sensor
    "p_channel_dropout": 0.15,
    "p_channel_scale": 0.3,
    "channel_scale_max": 0.05,
}
