import numpy as np
import matplotlib.pyplot as plt

from mocket.noise import generate_bl_noise
from mocket.workflows.paths import FIG_DIR


DT = 0.01
T_SIGNAL = 240.0
T_SHOW = 60.0
SEED = 13
BANDS = [0.005, 0.03, 0.1, 1.0]


def main():
    n = int(round(T_SIGNAL / DT))
    n_show = int(round(T_SHOW / DT))
    t = np.arange(n_show) * DT

    fig, axs = plt.subplots(len(BANDS), 1, figsize=(10, 8.2), sharex=True)

    for i, b in enumerate(BANDS):
        x = generate_bl_noise(b, T_SIGNAL, DT, seed=SEED + i)[:n_show]
        axs[i].plot(t, x, color="0.20", linewidth=1.0)
        axs[i].set_ylabel("Signal")
        axs[i].set_title(f"B = {b:g} Hz")
        axs[i].grid(True, alpha=0.22)

    axs[-1].set_xlabel("Time (s)")
    fig.suptitle("Figure 6 — Raw BL noise realizations across bandwidth regimes", y=0.995)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_06_noise_regimes_raw.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
