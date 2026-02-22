"""
Shared estimator registry used by evaluation/plot scripts.

Keeping method definitions in one place prevents drift between scripts.
"""

from mocket.estimators import (
    var_mean_bl_fit,
    var_mean_bl_std_fit,
    var_mean_bl_joint_fit,
    var_mean_bl_dual_fit,
)


METHODS = [
    ("BM", lambda w: var_mean_bl_fit(w, overlapping=False)),
    ("OBM", lambda w: var_mean_bl_fit(w, overlapping=True)),
    ("BM-std", lambda w: var_mean_bl_std_fit(w)),
    (
        "BM-joint",
        lambda w: var_mean_bl_joint_fit(w, sigma_clip_lo=0.8, sigma_clip_hi=1.25),
    ),
    ("BM-dual", lambda w: var_mean_bl_dual_fit(w)),
]


METHOD_STYLE = {
    "BM": ("tab:blue", "o", "-", 1.8),
    "OBM": ("tab:orange", "s", "--", 1.6),
    "BM-std": ("tab:green", "^", "-.", 1.8),
    "BM-joint": ("tab:purple", "*", "--", 2.0),
    "BM-dual": ("tab:cyan", "P", "-", 1.9),
}
