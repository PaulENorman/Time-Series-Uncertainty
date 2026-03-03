import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sfft
from scipy.signal import savgol_filter
from pathlib import Path
import os
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from noise import noise_from_pow_spec
    from paths import FIG_DIR
except ImportError:
    from mocket.noise import noise_from_pow_spec
    from mocket.workflows.paths import FIG_DIR

WORKFLOWS_DIR = Path(__file__).resolve().parent
if str(WORKFLOWS_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOWS_DIR))
from method_registry import HYBRID_METHODS, HYBRID_METHOD_STYLE

DT = 2e-4
BREAK_FREQ = 1.0
HI_FREQ_SLOPE = -5.0 / 3.0
B_FLAT = 10.0
SPEC_MODE = "piecewise"
TRANS_SHARPNESS = 8.0
T_SIGNAL = 300.0
T_GENERATE = 3.0 * T_SIGNAL
TIME_POINTS = np.logspace(np.log10(0.2), np.log10(100.0), 30)
N_TARGET_SIGNALS = 400
N_METHOD_SIGNALS = 400
N_WINDOWS_TARGET = 3
N_WINDOWS_METHOD = 3
BASE_SEED = 260219
KEEP_METHODS = list(HYBRID_METHODS.keys())
TRIM_FRACTION = 0.10
PLOT_SMOOTH = True
PLOT_SMOOTH_MAE = False
SMOOTH_WINDOW = 7
SMOOTH_POLY = 2


def _env_int(name, default):
    raw = os.getenv(name)
    return int(raw) if raw is not None else int(default)


def _env_float(name, default):
    raw = os.getenv(name)
    return float(raw) if raw is not None else float(default)


N_TARGET_SIGNALS = _env_int("TSU_N_TARGET", N_TARGET_SIGNALS)
N_METHOD_SIGNALS = _env_int("TSU_N_METHOD", N_METHOD_SIGNALS)
N_WINDOWS_TARGET = _env_int("TSU_NW_TARGET", N_WINDOWS_TARGET)
N_WINDOWS_METHOD = _env_int("TSU_NW_METHOD", N_WINDOWS_METHOD)
TRIM_FRACTION = _env_float("TSU_TRIM_FRAC", TRIM_FRACTION)
PLOT_SMOOTH = bool(_env_int("TSU_PLOT_SMOOTH", 1))
PLOT_SMOOTH_MAE = bool(_env_int("TSU_PLOT_SMOOTH_MAE", 0))
SMOOTH_WINDOW = _env_int("TSU_SMOOTH_WINDOW", SMOOTH_WINDOW)
SMOOTH_POLY = _env_int("TSU_SMOOTH_POLY", SMOOTH_POLY)
BREAK_FREQ = _env_float("TSU_BREAK_FREQ", BREAK_FREQ)
HI_FREQ_SLOPE = _env_float("TSU_HI_SLOPE", HI_FREQ_SLOPE)
B_FLAT = _env_float("TSU_B_FLAT", B_FLAT)
SPEC_MODE = os.getenv("TSU_SPEC_MODE", SPEC_MODE)
TRANS_SHARPNESS = _env_float("TSU_TRANS_SHARPNESS", TRANS_SHARPNESS)
TIME_POINTS = np.logspace(
    np.log10(0.2),
    np.log10(100.0),
    _env_int("TSU_N_TIME", len(TIME_POINTS)),
)


def _smooth_positive_curve(y):
    y = np.asarray(y, dtype=float)
    out = np.array(y, copy=True)
    valid = np.isfinite(y) & (y > 0)
    if not PLOT_SMOOTH or np.count_nonzero(valid) < 5:
        return out
    w = int(SMOOTH_WINDOW)
    if w % 2 == 0:
        w += 1
    w = max(5, min(w, np.count_nonzero(valid) if np.count_nonzero(valid) % 2 == 1 else np.count_nonzero(valid) - 1))
    if w < 5:
        return out
    p = min(int(SMOOTH_POLY), w - 2)
    ly = np.log(y[valid])
    ly_s = savgol_filter(ly, window_length=w, polyorder=max(1, p), mode="interp")
    out[valid] = np.exp(ly_s)
    return out


def _build_signal(seed):
    n = int(T_SIGNAL / DT)
    n_long = int(T_GENERATE / DT)
    freqs = sfft.rfftfreq(n_long, DT)
    p = np.zeros_like(freqs)
    if SPEC_MODE == "flat_cutoff":
        p[(freqs > 0.0) & (freqs <= B_FLAT)] = 1.0
    elif SPEC_MODE == "flat_smooth_tail":
        r = np.maximum(freqs / max(B_FLAT, 1e-12), 0.0)
        k = max(float(TRANS_SHARPNESS), 1e-6)
        p = np.power(1.0 + np.power(r, k), HI_FREQ_SLOPE / k)
        p[freqs <= 0.0] = 0.0
    else:
        p[:] = 1.0
        p[0] = 0.0
        mask_hi = freqs > BREAK_FREQ
        p[mask_hi] = np.power(freqs[mask_hi] / BREAK_FREQ, HI_FREQ_SLOPE)
    x_long = np.asarray(noise_from_pow_spec(freqs, p, seed=seed), dtype=float)
    rng = np.random.default_rng(seed + 5003)
    start = int(rng.integers(0, max(1, n_long - n)))
    x = x_long[start : start + n]
    x -= np.mean(x)
    s = float(np.std(x))
    return x / s if s > 0 else x


def _precompute_starts(n_signals, n_long, time_points, n_windows, seed):
    rng = np.random.default_rng(seed)
    starts_by_t = []
    for t in time_points:
        n_eval = int(round(float(t) / DT))
        max_start = max(1, n_long - n_eval)
        if max_start <= 1:
            starts = np.zeros((n_signals, n_windows), dtype=int)
        else:
            starts = rng.integers(0, max_start, size=(n_signals, n_windows), endpoint=False)
        starts_by_t.append(starts)
    return starts_by_t


def _robust_mean(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    if vals.size < 10 or TRIM_FRACTION <= 0.0:
        return float(np.mean(vals))
    vals = np.sort(vals)
    k = int(TRIM_FRACTION * vals.size)
    if 2 * k >= vals.size:
        return float(np.mean(vals))
    return float(np.mean(vals[k:-k]))


def _mc_target(signals, time_points, starts_by_t):
    n_long = len(signals[0])
    y = []
    for ti, t in enumerate(time_points):
        n_eval = int(round(float(t) / DT))
        max_start = max(1, n_long - n_eval)
        means, sigmas = [], []
        starts_all = starts_by_t[ti]
        for si, sig in enumerate(signals):
            starts = starts_all[si, :N_WINDOWS_TARGET]
            for s in starts:
                w = sig[s : s + n_eval]
                means.append(float(np.mean(w)))
                sigmas.append(float(np.std(w)))
        means = np.asarray(means)
        sigmas = np.asarray(sigmas)
        sm = float(np.std(means, ddof=1))
        ss = float(np.mean(sigmas))
        y.append(sm / max(ss, 1e-15))
    return np.asarray(y, dtype=float)


def _method_curves(signals, time_points, starts_by_t):
    estimators = {k: HYBRID_METHODS[k] for k in KEEP_METHODS if k in HYBRID_METHODS}
    n_long = len(signals[0])
    curves = {k: [] for k in estimators}
    for ti, t in enumerate(time_points):
        n_eval = int(round(float(t) / DT))
        max_start = max(1, n_long - n_eval)
        starts_all = starts_by_t[ti]
        for k, fn in estimators.items():
            vals = []
            for si, sig in enumerate(signals):
                starts = starts_all[si, :N_WINDOWS_METHOD]
                for s in starts:
                    w = sig[s : s + n_eval]
                    vs = float(np.var(w))
                    if vs <= 0:
                        continue
                    est = float(fn(w))
                    if np.isfinite(est) and est > 0:
                        vals.append(float(np.sqrt(est / vs)))
            curves[k].append(_robust_mean(vals))
    return {k: np.asarray(v) for k, v in curves.items()}


def _plot_figure_14(y_mc, preds):
    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    y_mc_plot = _smooth_positive_curve(y_mc)
    ax.loglog(TIME_POINTS, y_mc_plot, "k-", lw=2.2, label="MC target")
    for k in KEEP_METHODS:
        c, m, ls, _ = HYBRID_METHOD_STYLE[k]
        yk = _smooth_positive_curve(preds[k])
        ax.loglog(
            TIME_POINTS,
            yk,
            color=c,
            marker=m,
            ls=ls,
            lw=1.8,
            ms=4.2,
            label=k,
        )
    ax.set_xlabel("Integration time t (s)")
    ax.set_ylabel(r"$\sigma_{\mathrm{mean}}/\sigma_{\mathrm{signal}}$")
    if SPEC_MODE == "flat_cutoff":
        ax.set_title(f"Band-limited (B={B_FLAT:g} Hz) method comparison")
    elif SPEC_MODE == "flat_smooth_tail":
        ax.set_title(f"Flat-to-B={B_FLAT:g} Hz with smooth $f^{{-5/3}}$ roll-off")
    else:
        ax.set_title("Piecewise-spectrum method comparison")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    out = FIG_DIR / "figure_17_realworld_methods.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    return out


def _plot_figure_15(y_mc, preds):
    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    valid = np.isfinite(y_mc) & (y_mc > 0)
    t = TIME_POINTS[valid]
    y_mc_raw = y_mc[valid]
    y_mc_plot = _smooth_positive_curve(y_mc)[valid] if PLOT_SMOOTH_MAE else y_mc_raw
    for k in KEEP_METHODS:
        c, m, ls, _ = HYBRID_METHOD_STYLE[k]
        y_raw = preds[k][valid]
        y_plot = _smooth_positive_curve(preds[k])[valid] if PLOT_SMOOTH_MAE else y_raw
        mae_t = np.abs((y_plot - y_mc_plot) / y_mc_plot)
        ax.loglog(
            t,
            np.clip(mae_t, 1e-12, np.inf),
            color=c,
            marker=m,
            ls=ls,
            lw=1.8,
            ms=4.2,
            label=k,
        )
    ax.set_xlabel("Integration time t (s)")
    ax.set_ylabel("Absolute relative error vs MC target")
    if SPEC_MODE == "flat_cutoff":
        ax.set_title(f"Method error vs integration time (band-limited B={B_FLAT:g} Hz)")
    elif SPEC_MODE == "flat_smooth_tail":
        ax.set_title(f"Method error vs integration time (flat-to-{B_FLAT:g} Hz, smooth tail)")
    else:
        ax.set_title("Method error vs integration time (piecewise-spectrum signal)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    out = FIG_DIR / "figure_18_realworld_mae_vs_time.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    return out


def main():
    target = [_build_signal(BASE_SEED + 10000 + i) for i in range(N_TARGET_SIGNALS)]
    if N_METHOD_SIGNALS == N_TARGET_SIGNALS:
        method = target
    else:
        method = [_build_signal(BASE_SEED + 20000 + i) for i in range(N_METHOD_SIGNALS)]
    n_windows = max(N_WINDOWS_TARGET, N_WINDOWS_METHOD)
    starts_target = _precompute_starts(
        len(target), len(target[0]), TIME_POINTS, n_windows, BASE_SEED + 800001
    )
    y_mc = _mc_target(target, TIME_POINTS, starts_target)
    if method is target:
        starts_method = starts_target
    else:
        starts_method = _precompute_starts(
            len(method), len(method[0]), TIME_POINTS, n_windows, BASE_SEED + 800002
        )
    preds = _method_curves(method, TIME_POINTS, starts_method)
    out14 = _plot_figure_14(y_mc, preds)
    out15 = _plot_figure_15(y_mc, preds)
    print(out14)
    print(out15)


if __name__ == "__main__":
    main()
