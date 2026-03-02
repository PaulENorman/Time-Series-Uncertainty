import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from noise import generate_piecewise_power_noise
    from stabilization import (
        add_exponential_transient,
        estimate_stabilization_time_mockett,
    )
    from paths import FIG_DIR
except ImportError:
    from mocket.noise import generate_piecewise_power_noise
    from mocket.stabilization import (
        add_exponential_transient,
        estimate_stabilization_time_mockett,
    )
    from mocket.workflows.paths import FIG_DIR


DT = 2e-4
T_SIGNAL = 6.0
BREAK_FREQ = 10.0
TAIL_SLOPE = -5.0 / 3.0
TRANSIENT_CUTOFF_TIME = 1.0
TRANSIENT_AMP_STD = 2.0
SEED = 20260301


def _build_signal():
    t_generate = 3.0 * T_SIGNAL
    x_long = generate_piecewise_power_noise(
        break_freq=BREAK_FREQ,
        slope=TAIL_SLOPE,
        length=t_generate,
        dt=DT,
        seed=SEED,
    )
    n = int(round(T_SIGNAL / DT))
    rng = np.random.default_rng(SEED + 10)
    start = int(rng.integers(0, max(1, len(x_long) - n)))
    x = np.asarray(x_long[start : start + n], dtype=float)

    t = np.arange(n, dtype=float) * DT
    x_w_tr, transient = add_exponential_transient(
        x,
        dt=DT,
        amplitude_std=TRANSIENT_AMP_STD,
        cutoff_time=TRANSIENT_CUTOFF_TIME,
        cutoff_fraction=0.10,
    )
    return t, x_w_tr, transient


def main():
    t, x, transient = _build_signal()
    res = estimate_stabilization_time_mockett(
        x,
        dt=DT,
        max_remove_fraction=0.5,
        min_segment_samples=256,
    )
    t_remove, s_vals, t_star = res.t_remove, res.s_curve, res.t_star
    x_max = 0.5 * T_SIGNAL

    fig, axs = plt.subplots(2, 1, figsize=(10.0, 7.4), sharex=True)

    ax = axs[0]
    ax.plot(t, x, color="0.20", lw=1.0, label="Signal with transient")
    ax.plot(t, transient + np.mean(x), color="tab:red", lw=1.5, ls="--", label="Added transient")
    ax.axvline(TRANSIENT_CUTOFF_TIME, color="tab:red", ls=":", lw=1.2, label="True cutoff (90% decay)")
    if np.isfinite(t_star):
        ax.axvline(t_star, color="tab:blue", ls=":", lw=1.3, label=f"Detected transient end: {t_star:.2f} s")
    ax.set_xlim(0.0, x_max)
    ax.set_ylabel("Signal")
    ax.set_title("Synthetic signal with initial transient")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    ax = axs[1]
    ax.plot(t_remove, s_vals, color="tab:purple", lw=1.8, label=r"$s(t_{\mathrm{remove}})$ via ACC-0c")
    if np.isfinite(t_star):
        ax.axvline(t_star, color="tab:blue", ls=":", lw=1.3, label=f"Minimum at {t_star:.2f} s")
    ax.set_xlim(0.0, x_max)
    ax.set_xlabel("Time / removed-initial-time (s)")
    ax.set_ylabel(r"$s$ (estimated std. of mean)")
    ax.set_title("Mockett-style stationarity scan")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_16_stationarity_mockett.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
