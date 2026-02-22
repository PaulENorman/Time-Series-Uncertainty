import numpy as np
import matplotlib.pyplot as plt

from mocket.workflows.plot_eval import compute_curves
from mocket.workflows.paths import FIG_DIR


B_VALUES = [0.005, 0.008, 0.012, 0.03, 0.1, 1.0, 15.0, 40.0]


def main():
    bm_mae = []
    obm_mae = []

    for b in B_VALUES:
        analytic, preds = compute_curves(b)
        valid = np.isfinite(analytic) & (analytic > 0)
        a = analytic[valid]

        p_bm = preds["BM"][valid]
        p_obm = preds["OBM"][valid]
        bm_mae.append(float(np.nanmean(np.abs((p_bm - a) / a))))
        obm_mae.append(float(np.nanmean(np.abs((p_obm - a) / a))))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.array(B_VALUES, dtype=float)

    ax.semilogx(x, bm_mae, color="tab:blue", marker="o", linestyle="-", linewidth=2.0, markersize=6, label="BM")
    ax.semilogx(x, obm_mae, color="tab:orange", marker="s", linestyle="--", linewidth=2.0, markersize=6, label="OBM")

    ax.set_xlabel("Bandwidth B (Hz)")
    ax.set_ylabel("Mean absolute relative error vs analytic")
    ax.set_title("Figure 10 (BM+OBM) — Performance across bandwidth")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_10_bm_obm_performance_vs_B.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
