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
from methods_bm_obm_acc_tail import (
    var_mean_bm,
    var_mean_obm75,
    var_mean_acc_zero_corrected,
    var_mean_tail,
)
from plot_eval import compute_curves


B_VALUES = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
FIG12_METHODS = {
    "BM": var_mean_bm,
    "OBM 75%": var_mean_obm75,
    "ACC-0c": var_mean_acc_zero_corrected,
    "Tail": var_mean_tail,
}
FIG12_STYLE = {
    "BM": ("tab:blue", "o", "-", 1.9),
    "OBM 75%": ("tab:orange", "s", "--", 1.9),
    "ACC-0c": ("tab:purple", "D", ":", 1.9),
    "Tail": ("tab:green", "^", "-.", 2.0),
}


def main():
    maes = {k: [] for k in FIG12_METHODS}
    for b in B_VALUES:
        analytic, preds = compute_curves(b, estimators=FIG12_METHODS, t_generate=300.0, n_signals=8)
        valid = np.isfinite(analytic) & (analytic > 0)
        a = analytic[valid]
        for k in FIG12_METHODS:
            p = preds[k][valid]
            maes[k].append(float(np.nanmean(np.abs((p - a) / a))))

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    x = np.array(B_VALUES, dtype=float)
    for k in FIG12_METHODS:
        c, m, ls, _ = FIG12_STYLE[k]
        y = np.clip(np.array(maes[k], dtype=float), 1e-12, np.inf)
        ax.loglog(x, y, color=c, marker=m, ls=ls, lw=1.9, ms=5.0, label=k)
    ax.set_xlabel("Bandwidth B (Hz)")
    ax.set_ylabel("MAE vs analytic")
    ax.set_title("MAE across BL regimes")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    out = FIG_DIR / "figure_12_methods_mae_vs_B.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
