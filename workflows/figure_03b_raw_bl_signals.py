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
T_SHOW = 5.0
T_GEN = 3.0 * T_SHOW
B_VALUES = [0.1, 1.0, 10.0]
BASE_SEED = 20260303


def _build(B, seed):
    x_long = np.asarray(generate_bl_noise(B, T_GEN, DT, seed=seed), dtype=float)
    n = int(round(T_SHOW / DT))
    rng = np.random.default_rng(seed + 99)
    s0 = int(rng.integers(0, max(1, x_long.size - n)))
    x = x_long[s0 : s0 + n]
    x -= np.mean(x)
    return x


def main():
    fig, axs = plt.subplots(3, 1, figsize=(10.2, 7.2), sharex=True)
    t = np.arange(int(round(T_SHOW / DT))) * DT
    for ax, B in zip(axs, B_VALUES):
        x = _build(B, BASE_SEED + int(B * 1000))
        ax.plot(t, x, color="0.2", lw=0.9)
        ax.set_ylabel("Signal")
        ax.set_title(f"B = {B:g} Hz")
        ax.grid(True, alpha=0.25)
    axs[-1].set_xlabel("Time (s)")
    fig.suptitle("Raw bandwidth-limited signals across B (5 s windows)")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_03b_raw_bl_signals.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
