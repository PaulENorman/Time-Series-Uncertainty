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
T_GENERATE = 3.0 * T_SIGNAL
BREAK_FREQ = 10.0
TAIL_SLOPE = -5.0 / 3.0
TRANSIENT_CUTOFF_TIME = 1.0
AMP_LEVELS = [1.0, 2.0, 3.0]
N_SIGNALS = 420
BASE_SEED = 20260311


def _build_signal(seed, amp_std):
    x_long = generate_piecewise_power_noise(
        break_freq=BREAK_FREQ,
        slope=TAIL_SLOPE,
        length=T_GENERATE,
        dt=DT,
        seed=seed,
    )
    n = int(round(T_SIGNAL / DT))
    rng = np.random.default_rng(seed + 100)
    start = int(rng.integers(0, max(1, len(x_long) - n)))
    x = np.asarray(x_long[start : start + n], dtype=float)
    x_w_tr, _ = add_exponential_transient(
        x,
        dt=DT,
        amplitude_std=float(amp_std),
        cutoff_time=TRANSIENT_CUTOFF_TIME,
        cutoff_fraction=0.10,
    )
    return x, x_w_tr


def _smooth_density_line(samples, x_grid, bw_bins=2.0):
    hist, edges = np.histogram(samples, bins=x_grid, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Gaussian smoothing in bin space.
    radius = int(max(3, np.ceil(4.0 * bw_bins)))
    kx = np.arange(-radius, radius + 1, dtype=float)
    kern = np.exp(-0.5 * (kx / float(bw_bins)) ** 2)
    kern /= np.sum(kern)
    y = np.convolve(hist, kern, mode="same")
    return centers, y


def main():
    fig, axs = plt.subplots(2, 1, figsize=(9.4, 8.4), sharex=False)
    ax_top, ax_bot = axs
    bins_t = np.linspace(0.0, 2.2, 140)
    bins_b = np.linspace(-0.06, 0.26, 150)
    colors = {1.0: "tab:blue", 2.0: "tab:purple", 3.0: "tab:green"}

    for amp in AMP_LEVELS:
        t_star = []
        mean_bias = []
        for i in range(N_SIGNALS):
            x_base, x = _build_signal(BASE_SEED + 10000 * int(amp) + i, amp_std=amp)
            res = estimate_stabilization_time_mockett(
                x,
                dt=DT,
                max_remove_fraction=0.5,
                min_segment_samples=256,
            )
            if np.isfinite(res.t_star):
                t_star.append(float(res.t_star))
            mean_bias.append(float(np.mean(x) - np.mean(x_base)))
        t_star = np.asarray(t_star, dtype=float)
        mean_bias = np.asarray(mean_bias, dtype=float)
        if t_star.size == 0:
            continue
        xc, yc = _smooth_density_line(t_star, bins_t, bw_bins=2.0)
        ax_top.plot(xc, yc, color=colors[amp], lw=2.0, label=f"{int(amp)}$\\sigma$ transient")
        xb, yb = _smooth_density_line(mean_bias, bins_b, bw_bins=2.0)
        ax_bot.plot(xb, yb, color=colors[amp], lw=2.0, label=f"{int(amp)}$\\sigma$ transient")

    ax_top.axvline(TRANSIENT_CUTOFF_TIME, color="tab:red", ls="--", lw=1.6, label="True cutoff (1.0 s)")
    ax_top.set_xlabel("Predicted stabilization time (s)")
    ax_top.set_ylabel("Density")
    ax_top.set_title("Detected stabilization-time distributions (ACC-0c, Mockett scan)")
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(fontsize=9)

    ax_bot.axvline(0.0, color="0.35", ls="--", lw=1.2)
    ax_bot.set_xlabel("Added bias in sample mean")
    ax_bot.set_ylabel("Density")
    ax_bot.set_title("Distribution of transient-induced mean bias")
    ax_bot.grid(True, alpha=0.25)
    ax_bot.legend(fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_17_stationarity_distribution.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
