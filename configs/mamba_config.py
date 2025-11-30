"""
Mamba Configuration.

This module defines the configuration for the Mamba-based SensorModulator.
"""

MAMBA_CONFIG = {
    "d_state": 16,      # SSM state expansion factor
    "d_conv": 4,        # Local convolution width
    "expand": 2,        # Block expansion factor
    "use_mamba": True,  # Whether to use Mamba instead of GRU
}
