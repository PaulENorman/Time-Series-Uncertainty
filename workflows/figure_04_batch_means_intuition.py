import numpy as np
import matplotlib.pyplot as plt

from mocket.noise import generate_bl_noise
from mocket.estimators import bm_var_mean
from mocket.workflows.paths import FIG_DIR


DT = 2e-4
T_WINDOW = 6.0
B = 15.0
SEED = 22


def _panel_batches(ax, t, x, b, color):
    n = len(x)
    nb = n // b
    tt = t[: nb * b]
    xx = x[: nb * b]
    segs = xx.reshape(nb, b)
    means = np.mean(segs, axis=1)

    ax.plot(tt, xx, color="0.25", lw=1.0)
    for i in range(nb):
        s = i * b
        e = (i + 1) * b
        ts = tt[s]
        te = tt[e - 1]
        ax.hlines(means[i], ts, te, colors=color, linewidth=2.0)

    ax.set_xlim(tt[0], tt[-1])
    ax.set_ylabel("Signal")
    ax.grid(True, alpha=0.25)


def main():
    n = int(round(T_WINDOW / DT))
    x = generate_bl_noise(B, T_WINDOW, DT, seed=SEED)[:n]
    t = np.arange(n) * DT

    b_small = max(20, n // 36)
    b_large = max(20, n // 9)

    b_vals = np.unique(np.logspace(np.log10(max(20, n // 60)), np.log10(max(25, n // 6)), 18).astype(int))
    v_vals = np.array([bm_var_mean(x, int(b)) for b in b_vals], dtype=float)
    valid = np.isfinite(v_vals) & (v_vals > 0)
    b_vals = b_vals[valid]
    v_vals = v_vals[valid]

    v_small = bm_var_mean(x, b_small)
    v_large = bm_var_mean(x, b_large)

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.2], hspace=0.28)

    ax1 = fig.add_subplot(gs[0])
    _panel_batches(ax1, t, x, b_small, color="tab:blue")
    ax1.set_title(f"Figure 4A — Small batch size (b={b_small} samples): many local means")

    ax2 = fig.add_subplot(gs[1])
    _panel_batches(ax2, t, x, b_large, color="tab:orange")
    ax2.set_title(f"Figure 4B — Large batch size (b={b_large} samples): stronger averaging")
    ax2.set_xlabel("Time (s)")

    ax3 = fig.add_subplot(gs[2])
    x_sec = b_vals * DT
    y_std = np.sqrt(v_vals)
    ax3.loglog(x_sec, y_std, "o-", color="0.25", lw=1.4, ms=4, label="Observed batch-mean std")
    ax3.loglog(b_small * DT, np.sqrt(v_small), "o", color="tab:blue", ms=8, label=f"small b={b_small}")
    ax3.loglog(b_large * DT, np.sqrt(v_large), "o", color="tab:orange", ms=8, label=f"large b={b_large}")

    t_end = T_WINDOW
    t0 = float(max(x_sec[0], 1e-9))
    y0 = float(y_std[0])
    t_ref = np.logspace(np.log10(t0), np.log10(t_end), 150)
    y_ref = y0 * (t_ref / t0) ** (-0.5)
    ax3.loglog(t_ref, y_ref, "--", color="tab:red", lw=1.8, label=r"$t^{-1/2}$ extrapolation")
    ax3.axvline(t_end, color="tab:red", linestyle=":", lw=1.2, alpha=0.8)
    ax3.set_xlabel("Batch length b (seconds)")
    ax3.set_ylabel(r"$\sigma_{\mathrm{batch\ mean}}$")
    ax3.set_title(f"Figure 4C — Scaling curve with $t^{{-1/2}}$ extrapolation to $T_{{end}}$ (B={B:g} Hz)")
    ax3.grid(True, which="both", alpha=0.25)
    ax3.legend(fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_04_batch_means_intuition.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
