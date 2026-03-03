"""Stationarity and stabilization-time utilities for finite time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    from estimators import var_mean_acc_unbiased_bl_corrected
except ImportError:
    from .estimators import var_mean_acc_unbiased_bl_corrected


EstimatorFn = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class StabilizationResult:
    """Container for stabilization-time scan outputs."""

    t_star: float
    i_star: int
    t_remove: np.ndarray
    s_curve: np.ndarray


def add_exponential_transient(
    signal: np.ndarray,
    dt: float,
    amplitude_std: float = 2.0,
    cutoff_time: float = 1.0,
    cutoff_fraction: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add an exponentially decaying transient with a hard cutoff.

    The transient starts at `amplitude_std * std(signal)` and is forced to
    exactly zero once it decays to `cutoff_fraction` of its initial value.
    """
    x = np.asarray(signal, dtype=float)
    n = x.size
    t = np.arange(n, dtype=float) * float(dt)

    sigma = float(np.std(x))
    amp0 = float(amplitude_std) * sigma
    if amp0 <= 0:
        return x.copy(), np.zeros_like(x)

    frac = float(np.clip(cutoff_fraction, 1e-8, 0.5))
    alpha = -np.log(frac) / max(float(cutoff_time), 1e-12)
    # Shift/subtract the cutoff fraction so the hard cutoff is continuous.
    # This preserves the initial amplitude at t=0 and removes the step jump
    # when forcing the tail to zero.
    exp_term = np.exp(-alpha * t)
    transient = amp0 * (exp_term - frac) / max(1.0 - frac, 1e-12)
    transient = np.maximum(transient, 0.0)
    return x + transient, transient


def _eval_s_for_indices(
    x: np.ndarray,
    idx: np.ndarray,
    estimator: EstimatorFn,
    min_segment_samples: int,
) -> np.ndarray:
    s = np.full(idx.size, np.nan, dtype=float)
    n = x.size
    for i, i0 in enumerate(idx):
        i0 = int(i0)
        if i0 < 0:
            i0 = 0
        if i0 >= n:
            continue
        seg = x[i0:]
        if seg.size < int(min_segment_samples):
            continue
        v = float(estimator(seg))
        if np.isfinite(v) and v > 0:
            s[i] = float(np.sqrt(v))
    return s


def estimate_stabilization_time_mockett(
    signal: np.ndarray,
    dt: float,
    estimator: EstimatorFn = var_mean_acc_unbiased_bl_corrected,
    max_remove_fraction: float = 0.5,
    coarse_points: int = 48,
    refine_points: int = 48,
    refine_half_width: float = 0.08,
    min_segment_samples: int = 256,
) -> StabilizationResult:
    """
    Estimate stabilization time using a fast Mockett-style scan.

    Speed optimization:
    - coarse global scan across [0, max_remove_fraction * T]
    - local refinement around coarse minimum
    """
    x = np.asarray(signal, dtype=float)
    n = x.size
    if n < max(int(min_segment_samples), 16):
        return StabilizationResult(np.nan, -1, np.array([], dtype=float), np.array([], dtype=float))

    max_i = int(np.clip(max_remove_fraction, 0.05, 0.95) * n)
    max_i = max(1, min(max_i, n - int(min_segment_samples)))

    coarse_idx = np.unique(np.linspace(0, max_i, int(max(8, coarse_points))).astype(int))
    s_coarse = _eval_s_for_indices(x, coarse_idx, estimator, min_segment_samples)
    valid_coarse = np.isfinite(s_coarse)
    if not np.any(valid_coarse):
        t_remove = coarse_idx.astype(float) * float(dt)
        return StabilizationResult(np.nan, -1, t_remove, s_coarse)

    i_best_c = int(np.nanargmin(s_coarse))
    i0 = int(coarse_idx[i_best_c])
    half = int(max(2, refine_half_width * max_i))
    lo = max(0, i0 - half)
    hi = min(max_i, i0 + half)

    refine_idx = np.unique(np.linspace(lo, hi, int(max(8, refine_points))).astype(int))
    s_refine = _eval_s_for_indices(x, refine_idx, estimator, min_segment_samples)
    valid_refine = np.isfinite(s_refine)
    if not np.any(valid_refine):
        t_remove = coarse_idx.astype(float) * float(dt)
        return StabilizationResult(np.nan, -1, t_remove, s_coarse)

    i_best_r = int(np.nanargmin(s_refine))
    i_star = int(refine_idx[i_best_r])
    t_star = float(i_star) * float(dt)

    # Return combined, sorted curve using already-evaluated points.
    s_map = {int(i): float(v) for i, v in zip(coarse_idx, s_coarse) if np.isfinite(v)}
    for i, v in zip(refine_idx, s_refine):
        if np.isfinite(v):
            s_map[int(i)] = float(v)
    all_idx = np.array(sorted(s_map.keys()), dtype=int)
    s_all = np.array([s_map[int(i)] for i in all_idx], dtype=float)
    t_remove = all_idx.astype(float) * float(dt)
    return StabilizationResult(t_star=t_star, i_star=i_star, t_remove=t_remove, s_curve=s_all)
