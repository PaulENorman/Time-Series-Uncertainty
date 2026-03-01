import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from estimators import (
    var_mean_acc_unbiased,
    var_mean_acc_unbiased_bl_corrected,
    var_mean_acc_tail_damped,
)
from plot_eval import compute_curves
from paths import FIG_DIR

ESTIMATORS = {
    "ACC-0": var_mean_acc_unbiased,
    "ACC-0c": var_mean_acc_unbiased_bl_corrected,
    "ACC-tail damp": var_mean_acc_tail_damped,
}

B_VALUES = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def main():
    maes = {k: [] for k in ESTIMATORS}
    for b in B_VALUES:
        analytic, preds = compute_curves(b, estimators=ESTIMATORS, t_generate=600.0, n_signals=12)
        valid = np.isfinite(analytic) & (analytic > 0)
        a = analytic[valid]
        for k in ESTIMATORS:
            p = preds[k][valid]
            maes[k].append(float(np.nanmean(np.abs((p - a) / a))))

    styles = {
        "ACC-0": ("tab:purple", "D", "-."),
        "ACC-0c": ("tab:green", "^", "-"),
        "ACC-tail damp": ("tab:orange", "s", "--"),
    }

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    x = np.array(B_VALUES, dtype=float)
    for k in ESTIMATORS:
        c, m, ls = styles[k]
        y = np.clip(np.array(maes[k], dtype=float), 1e-12, np.inf)
        ax.loglog(x, y, color=c, marker=m, ls=ls, lw=1.8, ms=4.6, label=k)
    ax.set_xlabel("Bandwidth B (Hz)")
    ax.set_ylabel("MAE vs analytic")
    ax.set_title("ACC variant MAE across B")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_08c_acc_variants_mae_vs_B.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
