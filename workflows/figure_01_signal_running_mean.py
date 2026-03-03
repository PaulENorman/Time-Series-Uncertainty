import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from noise import generate_bl_noise
from paths import FIG_DIR

DT = 2e-4
B = 15.0
T_SIGNAL = 60.0
T_GENERATE = 3.0 * T_SIGNAL
SEED = 7


def main():
    x_long = np.asarray(generate_bl_noise(B, T_GENERATE, DT, seed=SEED), dtype=float)
    n = int(round(T_SIGNAL / DT))
    rng = np.random.default_rng(SEED + 1001)
    start = int(rng.integers(0, max(1, x_long.size - n)))
    x = x_long[start : start + n]
    x = np.asarray(x, dtype=float)
    t = np.arange(x.size) * DT
    run = np.cumsum(x) / np.arange(1, x.size + 1)

    fig, axs = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True)
    axs[0].plot(t, x, color="tab:blue", lw=1.1)
    axs[0].set_ylabel("Signal")
    axs[0].set_title(f"Correlated signal realization (B={B:g})")
    axs[0].grid(True, alpha=0.25)

    axs[1].plot(t, run, color="tab:red", lw=1.6, label="Forward cumulative mean")
    mu_final = float(np.mean(x))
    axs[1].axhline(mu_final, color="k", ls="--", lw=1.1, label="Final mean")
    # Zoom around the final mean to show the uncertainty envelope clearly.
    lo = np.percentile(run, 5.0)
    hi = np.percentile(run, 95.0)
    pad = 0.08 * max(hi - lo, 1e-9)
    axs[1].set_ylim(lo - pad, hi + pad)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Running mean")
    axs[1].set_title("Signal and running mean")
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_01_signal_running_mean.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
