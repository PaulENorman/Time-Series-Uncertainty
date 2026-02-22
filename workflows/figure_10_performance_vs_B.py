import numpy as np
import matplotlib.pyplot as plt

from mocket.workflows.plot_eval import compute_curves
from mocket.workflows.paths import FIG_DIR


B_VALUES = [0.005, 0.008, 0.012, 0.03, 0.1, 1.0, 15.0, 40.0]


def main():
    bm_mae = []

    for b in B_VALUES:
        analytic, preds = compute_curves(b)
        valid = np.isfinite(analytic) & (analytic > 0)
        p = preds["BM"][valid]
        a = analytic[valid]
        mae = float(np.nanmean(np.abs((p - a) / a)))
        bm_mae.append(mae)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.array(B_VALUES, dtype=float)

    ax.semilogx(
        x,
        bm_mae,
        color="tab:blue",
        marker="o",
        linestyle="-",
        linewidth=2.0,
        markersize=7,
        alpha=0.95,
        label="BM",
    )

    ax.axvspan(0.004, 0.01, alpha=0.08, color="tab:red")
    ax.axvspan(0.01, 0.03, alpha=0.08, color="tab:orange")
    ax.axvspan(0.03, 1.0, alpha=0.08, color="tab:green")
    ax.axvspan(1.0, 60, alpha=0.08, color="tab:purple")

    ax.set_xlabel("Bandwidth B (Hz)")
    ax.set_ylabel("Mean absolute relative error vs analytic")
    ax.set_title("Figure 10 — BM performance across bandwidth")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_10_performance_vs_B.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
