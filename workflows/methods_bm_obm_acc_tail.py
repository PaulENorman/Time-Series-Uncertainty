import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from estimators import (
        bm_var_mean,
        var_mean_bl_fit,
        var_mean_acc_unbiased,
        var_mean_acc_unbiased_bl_corrected,
    )
except ImportError:
    from mocket.estimators import (
        bm_var_mean,
        var_mean_bl_fit,
        var_mean_acc_unbiased,
        var_mean_acc_unbiased_bl_corrected,
    )

DT = 2e-4


def var_mean_bm(signal):
    return float(var_mean_bl_fit(signal, overlapping=False))


def var_mean_obm50(signal):
    return float(var_mean_bl_fit(signal, overlapping=True, overlap_ratio=0.5))


def var_mean_obm75(signal):
    return float(var_mean_bl_fit(signal, overlapping=True, overlap_ratio=0.75))


def var_mean_acc_zero_corrected(signal):
    return float(var_mean_acc_unbiased_bl_corrected(signal))


def var_mean_acc_zero(signal):
    return float(var_mean_acc_unbiased(signal))

def _batch_curve(signal, min_b=20, fit_b=10):
    signal = np.asarray(signal, dtype=float)
    n = signal.size
    var_sig = float(np.var(signal))
    if n < 20 or var_sig <= 0:
        return None
    bl_end = int(n / min_b)
    if bl_end < 2:
        return None
    var_est = bm_var_mean(signal, bl_end) / max(var_sig, 1e-15)
    b0 = max(1e-12, 1.0 / (2.0 * max(var_est, 1e-12) * bl_end))
    bl_start = max(2, int(1.0 / (2.0 * b0)))
    batches = np.logspace(np.log10(bl_start), np.log10(bl_end), fit_b).astype(int)
    batches = np.unique(batches[(batches >= 2) & (batches <= bl_end)])
    if batches.size < 4:
        return None
    varb = np.array([bm_var_mean(signal, int(b)) for b in batches], dtype=float)
    valid = np.isfinite(varb) & (varb > 0)
    if valid.sum() < 4:
        return None
    b = batches[valid].astype(float)
    y = np.sqrt(varb[valid] / max(var_sig, 1e-15))
    return b * DT, y, var_sig


def _tail_conservative_from_curve(t_sec, y, var_sig, n_samples):
    k = min(6, len(t_sec))
    c_env = float(np.max(y[-k:] * np.sqrt(t_sec[-k:])))
    t_end = max(float(n_samples * DT), 1e-12)
    y_pred = float(c_env / np.sqrt(t_end))
    return float((max(y_pred, 1e-12) ** 2) * var_sig)


def var_mean_tail(signal):
    out = _batch_curve(signal)
    if out is None:
        return float(var_mean_bm(signal))
    t_sec, y, var_sig = out
    k = min(12, len(t_sec))
    if k < 6:
        return float(var_mean_bm(signal))

    # Fit tail in log-space with higher weight on later points.
    x = np.log(np.clip(t_sec[-k:], 1e-12, None))
    z = np.log(np.clip(y[-k:], 1e-15, None))
    w = np.linspace(0.7, 1.5, k)

    # Soft prior on slope toward -1/2 using a pseudo-observation.
    x_mean = float(np.mean(x))
    slope_prior = -0.5
    prior_strength = 1.3
    x_aug = np.append(x, x_mean + 1.0)
    z_aug = np.append(z, (x_mean + 1.0) * slope_prior)
    w_aug = np.append(w, prior_strength)

    slope_raw, intercept_raw = np.polyfit(x_aug, z_aug, 1, w=w_aug)
    slope = float(np.clip(slope_raw, -0.5, -0.05))

    # Stability gate: fallback to conservative envelope if tail is noisy/non-monotone.
    local_slopes = np.diff(z) / np.maximum(np.diff(x), 1e-12)
    slope_std = float(np.std(local_slopes))
    slope_mean = float(np.mean(local_slopes))
    if (slope_std > 0.30) or (slope_mean > -0.12) or (slope_mean < -0.9) or (slope_raw > -0.05):
        return _tail_conservative_from_curve(t_sec, y, var_sig, len(signal))

    # Re-anchor amplitude from median residual at fitted slope.
    intercept = float(np.median(z - slope * x))

    t_ref = float(t_sec[-2])
    y_ref = float(np.exp(slope * np.log(max(t_ref, 1e-12)) + intercept))
    t_end = max(float(len(signal) * DT), 1e-12)
    if (t_end / max(float(t_sec[-1]), 1e-12)) > 6.0:
        return _tail_conservative_from_curve(t_sec, y, var_sig, len(signal))
    y_pred = float(y_ref * (t_end / t_ref) ** slope)

    # Final sanity fallback.
    if (not np.isfinite(y_pred)) or (y_pred <= 0):
        return _tail_conservative_from_curve(t_sec, y, var_sig, len(signal))
    return float((max(y_pred, 1e-12) ** 2) * var_sig)
