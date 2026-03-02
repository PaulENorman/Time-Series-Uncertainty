import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from paths import FIG_DIR
except ImportError:
    from mocket.workflows.paths import FIG_DIR

WORKFLOWS_DIR = Path(__file__).resolve().parent
if str(WORKFLOWS_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOWS_DIR))
from methods_bm_obm_acc_tail import var_mean_bm, var_mean_tail
from plot_eval import compute_curves


TAIL_METHODS = {
    "BM": var_mean_bm,
    "Tail": var_mean_tail,
}

TAIL_STYLE = {
    "BM": ("tab:blue", "o", "-", 1.9),
    "Tail": ("tab:green", "^", "-.", 1.9),
}

B_VALUES = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def main():
    maes = {k: [] for k in TAIL_METHODS}
    for b in B_VALUES:
        analytic, preds = compute_curves(b, estimators=TAIL_METHODS, t_generate=600.0, n_signals=12)
        valid = np.isfinite(analytic) & (analytic > 0)
        a = analytic[valid]
        for k in TAIL_METHODS:
            p = preds[k][valid]
            maes[k].append(float(np.nanmean(np.abs((p - a) / a))))

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    x = np.array(B_VALUES, dtype=float)
    for k in TAIL_METHODS:
        c, m, ls, lw = TAIL_STYLE[k]
        y = np.clip(np.array(maes[k], dtype=float), 1e-12, np.inf)
        ax.loglog(x, y, color=c, marker=m, ls=ls, lw=lw, ms=5.0, label=k)
    ax.set_xlabel("Bandwidth B (Hz)")
    ax.set_ylabel("MAE vs analytic")
    ax.set_title("Tail method vs BM: MAE across B")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    out = FIG_DIR / "figure_09c_tail_vs_bm_mae.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
