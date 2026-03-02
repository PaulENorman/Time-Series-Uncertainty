import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sfft
from pathlib import Path
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
T_SIGNAL = 300.0
T_GENERATE = 3.0 * T_SIGNAL
TIME_POINTS = np.logspace(np.log10(0.2), np.log10(100.0), 20)
N_TARGET_SIGNALS = 800
N_METHOD_SIGNALS = 120
N_WINDOWS_TARGET = 3
N_WINDOWS_METHOD = 3
BASE_SEED = 260219
KEEP_METHODS = list(HYBRID_METHODS.keys())


def _build_signal(seed):
    n = int(T_SIGNAL / DT)
    n_long = int(T_GENERATE / DT)
    freqs = sfft.rfftfreq(n_long, DT)
    p = np.ones_like(freqs)
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


def _mc_target(signals, time_points, seed):
    n_long = len(signals[0])
    rng = np.random.default_rng(seed)
    y = []
    for t in time_points:
        n_eval = int(round(float(t) / DT))
        max_start = max(1, n_long - n_eval)
        means, sigmas = [], []
        for sig in signals:
            for _ in range(N_WINDOWS_TARGET):
                s = int(rng.integers(0, max_start))
                w = sig[s : s + n_eval]
                means.append(float(np.mean(w)))
                sigmas.append(float(np.std(w)))
        means = np.asarray(means)
        sigmas = np.asarray(sigmas)
        sm = float(np.std(means, ddof=1))
        ss = float(np.mean(sigmas))
        y.append(sm / max(ss, 1e-15))
    return np.asarray(y)


def _method_curves(signals, time_points, seed):
    estimators = {k: HYBRID_METHODS[k] for k in KEEP_METHODS if k in HYBRID_METHODS}
    n_long = len(signals[0])
    rng = np.random.default_rng(seed)
    curves = {k: [] for k in estimators}
    for t in time_points:
        n_eval = int(round(float(t) / DT))
        max_start = max(1, n_long - n_eval)
        for k, fn in estimators.items():
            vals = []
            for sig in signals:
                for _ in range(N_WINDOWS_METHOD):
                    s = int(rng.integers(0, max_start))
                    w = sig[s : s + n_eval]
                    vs = float(np.var(w))
                    if vs <= 0:
                        continue
                    est = float(fn(w))
                    if np.isfinite(est) and est > 0:
                        vals.append(float(np.sqrt(est / vs)))
            curves[k].append(float(np.mean(vals)) if vals else np.nan)
    return {k: np.asarray(v) for k, v in curves.items()}


def main():
    target = [_build_signal(BASE_SEED + 10000 + i) for i in range(N_TARGET_SIGNALS)]
    method = [_build_signal(BASE_SEED + 20000 + i) for i in range(N_METHOD_SIGNALS)]
    y_mc = _mc_target(target, TIME_POINTS, BASE_SEED + 300000)
    preds = _method_curves(method, TIME_POINTS, BASE_SEED + 400000)

    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    valid = np.isfinite(y_mc) & (y_mc > 0)
    t = TIME_POINTS[valid]
    for k in KEEP_METHODS:
        c, m, ls, _ = HYBRID_METHOD_STYLE[k]
        y = preds[k][valid]
        mae_t = np.abs((y - y_mc[valid]) / y_mc[valid])
        ax.loglog(t, np.clip(mae_t, 1e-12, np.inf), color=c, marker=m, ls=ls, lw=1.8, ms=4.2, label=k)
    ax.set_xlabel("Integration time t (s)")
    ax.set_ylabel("Absolute relative error vs MC target")
    ax.set_title("Method error vs integration time (piecewise-spectrum signal)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    out = FIG_DIR / "figure_15_realworld_mae_vs_time.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
