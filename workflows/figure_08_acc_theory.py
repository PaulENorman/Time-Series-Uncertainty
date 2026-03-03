import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sfft
from scipy.fft import next_fast_len
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from noise import generate_bl_noise
    from paths import FIG_DIR
except ImportError:
    from mocket.noise import generate_bl_noise
    from mocket.workflows.paths import FIG_DIR

DT = 2e-4


def _autocov_fft(x, max_lag):
    x = np.asarray(x, dtype=float)
    n = x.size
    x = x - np.mean(x)
    nfft = next_fast_len(2 * n - 1)
    fx = sfft.rfft(x, n=nfft)
    pxx = fx.real * fx.real + fx.imag * fx.imag
    c_full = sfft.irfft(pxx, n=nfft)
    c = c_full[:n]
    denom = np.arange(n, 0, -1, dtype=float)  # unbiased
    c = c / np.maximum(denom, 1.0)
    max_lag = min(int(max_lag), n - 1)
    return c[: max_lag + 1]


def main():
    # Use a noisier BL signal to make the zero-crossing truncation visible.
    b = 10.0
    t_sig = 6.0
    t_generate = 3.0 * t_sig
    x_long = np.asarray(generate_bl_noise(b, t_generate, DT, seed=20260228), dtype=float)
    n = int(round(t_sig / DT))
    rng = np.random.default_rng(20260229)
    start = int(rng.integers(0, max(1, x_long.size - n)))
    x = x_long[start : start + n]
    n = len(x)
    max_lag = int(0.2 * n)
    c = _autocov_fft(x, max_lag=max_lag)
    c = c / max(float(c[0]), 1e-15)
    tau = np.arange(c.size, dtype=float) / float(c.size - 1)
    c_damped = (1.0 - tau) * c
    zc_idx = np.where(c[1:] <= 0.0)[0]
    zc_idx = int(zc_idx[0] + 1) if zc_idx.size else len(tau) - 1
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 4.8))
    # Single heavy line showing the zero-crossing truncation behavior.
    c_used = np.full_like(c, np.nan)
    c_used[: zc_idx + 1] = c[: zc_idx + 1]
    ax.plot(
        tau,
        c_used,
        color="k",
        lw=3.0,
        label=r"$C_{xx}(\tau)$ used in ACC-zero",
    )
    ax.plot(
        tau,
        c_damped,
        color="tab:orange",
        lw=1.8,
        label=r"Damped tail $(1-\tau/T)C_{xx}(\tau)$",
    )
    ax.axvline(tau[zc_idx], color="tab:red", lw=1.3, ls=":", label="first zero crossing")
    ax.axhline(0, color="0.35", lw=1.0)
    ax.set_xlabel(r"Normalized lag $\tau/T$")
    ax.set_ylabel(r"Normalized autocovariance $C_{xx}(\tau)/C_{xx}(0)$")
    ax.set_title("ACC-zero and ACC-tail-damp behavior (B=10 signal)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    out = FIG_DIR / "figure_08_acc_theory.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
