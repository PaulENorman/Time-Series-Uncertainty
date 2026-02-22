import matplotlib.pyplot as plt
import numpy as np

from mocket.noise import generate_bl_noise
from mocket.workflows.paths import FIG_DIR


B_TRUE = 1.0
DT = 2e-4
T_SIGNAL = 60.0
SEED = 7


def main():
    signal = generate_bl_noise(B_TRUE, T_SIGNAL, DT, seed=SEED)
    if not np.all(np.isfinite(signal)):
        raise RuntimeError("Figure 0 signal generation failed: non-finite values. Increase duration or bandwidth.")
    t = np.arange(signal.size) * DT
    running_mean = np.cumsum(signal) / np.arange(1, signal.size + 1)

    fig, axs = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True)

    axs[0].plot(t, signal, color="tab:blue", linewidth=1.2)
    axs[0].set_ylabel("Signal value")
    axs[0].set_title("Figure 0 — One correlated-noise realization (B=1.0)")
    axs[0].grid(True, alpha=0.25)

    axs[1].plot(t, running_mean, color="tab:red", linewidth=1.8, label="Forward cumulative mean")
    axs[1].axhline(np.mean(signal), color="k", linestyle="--", linewidth=1.2, label="Final mean")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Cumulative mean")
    axs[1].set_title("How uncertain is this mean?")
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_00_signal_and_running_mean.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
