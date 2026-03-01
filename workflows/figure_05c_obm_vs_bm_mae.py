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
from method_registry import OBM_SWEEP_METHODS, OBM_SWEEP_STYLE
from plot_eval import compute_curves


B_VALUES = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def main():
    maes = {k: [] for k in OBM_SWEEP_METHODS}
    for b in B_VALUES:
        analytic, preds = compute_curves(b, estimators=OBM_SWEEP_METHODS, t_generate=600.0, n_signals=12)
        valid = np.isfinite(analytic) & (analytic > 0)
        a = analytic[valid]
        for k in OBM_SWEEP_METHODS:
            p = preds[k][valid]
            maes[k].append(float(np.nanmean(np.abs((p - a) / a))))

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    x = np.array(B_VALUES, dtype=float)
    for k in OBM_SWEEP_METHODS:
        c, m, ls, _ = OBM_SWEEP_STYLE[k]
        y = np.clip(np.array(maes[k], dtype=float), 1e-12, np.inf)
        ax.loglog(x, y, color=c, marker=m, ls=ls, lw=1.9, ms=5.0, label=k)
    ax.set_xlabel("Bandwidth B (Hz)")
    ax.set_ylabel("MAE vs analytic")
    ax.set_title("OBM overlap sweep MAE across B")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    out = FIG_DIR / "figure_07_obm_vs_bm_mae.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
