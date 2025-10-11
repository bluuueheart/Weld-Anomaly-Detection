"""Dataset configuration.

This module contains dataset-level constants and backward-compatible aliases
for older code and tests.
"""

# Data paths
DATA_ROOT = "Data"

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
