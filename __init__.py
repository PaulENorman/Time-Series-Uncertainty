"""
MOCKET – Variance-of-the-mean estimators and band-limited noise generation.

Core estimators
---------------
bm_var_mean          – non-overlapping batch-means variance
obm_mean_var         – overlapping batch-means variance
var_mean_bl_fit      – BL-fit via batch means (overlapping or non-overlapping)

Noise generation
----------------
band_limited_noise        – bandwidth-limited white noise
generate_bl_noise         – convenience wrapper (passband 0…B Hz)
generate_multiband_noise  – sum of independent BL components, unit variance
"""

from .estimators import (
    bm_var_mean,
    obm_mean_var,
    var_mean_bl,
    var_bl,
    log_var_mean_bl,
    var_mean_bl_fit,
    var_mean_bl_std_fit,
    var_mean_bl_joint_fit,
)

from .noise import (
    band_limited_noise,
    generate_bl_noise,
    generate_multiband_noise,
    noise_from_pow_spec,
)

from .stabilization import (
    StabilizationResult,
    add_exponential_transient,
    estimate_stabilization_time_mockett,
)

__all__ = [
    "bm_var_mean",
    "obm_mean_var",
    "var_mean_bl",
    "var_bl",
    "log_var_mean_bl",
    "var_mean_bl_fit",
    "var_mean_bl_std_fit",
    "var_mean_bl_joint_fit",
    "band_limited_noise",
    "generate_bl_noise",
    "generate_multiband_noise",
    "noise_from_pow_spec",
    "StabilizationResult",
    "add_exponential_transient",
    "estimate_stabilization_time_mockett",
]
