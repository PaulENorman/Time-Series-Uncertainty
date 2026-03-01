import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from noise import generate_bl_noise
from estimators import bm_var_mean
from paths import FIG_DIR

DT = 2e-4
T_WINDOW = 6.0
T_GENERATE = 3.0 * T_WINDOW
B = 15.0
SEED = 22


def _panel_batches(ax, t, x, b, color):
    nb = len(x) // b
    tt = t[: nb * b]
    xx = x[: nb * b]
    segs = xx.reshape(nb, b)
    means = np.mean(segs, axis=1)
    ax.plot(tt, xx, color="0.25", lw=1.0)
    for i in range(nb):
        s = i * b
        e = (i + 1) * b
        ax.hlines(means[i], tt[s], tt[e - 1], colors=color, lw=1.9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(tt[0], tt[-1])


def main():
    n = int(round(T_WINDOW / DT))
    x_long = np.asarray(generate_bl_noise(B, T_GENERATE, DT, seed=SEED), dtype=float)
    rng = np.random.default_rng(SEED + 1002)
    start = int(rng.integers(0, max(1, x_long.size - n)))
    x = x_long[start : start + n]
    t = np.arange(n) * DT

    b_small = max(20, n // 36)
    b_large = max(20, n // 9)
    b_vals = np.unique(np.logspace(np.log10(max(20, n // 60)), np.log10(max(25, n // 6)), 18).astype(int))
    v_vals = np.array([bm_var_mean(x, int(b)) for b in b_vals], dtype=float)
    good = np.isfinite(v_vals) & (v_vals > 0)
    b_vals = b_vals[good]
    v_vals = v_vals[good]

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.2], hspace=0.28)
    ax1 = fig.add_subplot(gs[0])
    _panel_batches(ax1, t, x, b_small, "tab:blue")
    ax1.set_ylabel("Signal")
    ax1.set_title(f"Small batch size (b={b_small} samples)")

    ax2 = fig.add_subplot(gs[1])
    _panel_batches(ax2, t, x, b_large, "tab:orange")
    ax2.set_ylabel("Signal")
    ax2.set_xlabel("Time (s)")
    ax2.set_title(f"Large batch size (b={b_large} samples)")

    ax3 = fig.add_subplot(gs[2])
    x_sec = b_vals * DT
    y_std = np.sqrt(v_vals)
    ax3.loglog(x_sec, y_std, "o-", color="0.25", ms=4, lw=1.4, label="Observed")
    # Highlight the two batch sizes shown in panels 1 and 2.
    i_small = int(np.argmin(np.abs(b_vals - b_small)))
    i_large = int(np.argmin(np.abs(b_vals - b_large)))
    ax3.loglog(
        [x_sec[i_small]],
        [y_std[i_small]],
        marker="o",
        color="tab:blue",
        ms=8,
        lw=0,
        label=f"Small-batch point (b={b_vals[i_small]})",
    )
    ax3.loglog(
        [x_sec[i_large]],
        [y_std[i_large]],
        marker="o",
        color="tab:orange",
        ms=8,
        lw=0,
        label=f"Large-batch point (b={b_vals[i_large]})",
    )
    tref = np.logspace(np.log10(max(x_sec[0], 1e-6)), np.log10(T_WINDOW), 120)
    yref = y_std[0] * (tref / max(x_sec[0], 1e-9)) ** (-0.5)
    ax3.loglog(tref, yref, "--", color="tab:red", lw=1.7, label=r"$t^{-1/2}$ guide")
    ax3.set_xlabel("Batch length (s)")
    ax3.set_ylabel(r"$\sigma_{\mathrm{batch\ mean}}$")
    ax3.set_title("Batch-means intuition")
    ax3.grid(True, which="both", alpha=0.25)
    ax3.legend(fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_04_bm_intuition.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
