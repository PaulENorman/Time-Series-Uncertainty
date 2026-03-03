import matplotlib.pyplot as plt
import numpy as np
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
from plot_eval import TIME_POINTS, compute_curves
from paths import FIG_DIR

ESTIMATORS = {
    "ACC-0": var_mean_acc_unbiased,
    "ACC-0c": var_mean_acc_unbiased_bl_corrected,
    "ACC-tail damp": var_mean_acc_tail_damped,
}


def main():
    b = 1.0
    analytic, preds = compute_curves(b, estimators=ESTIMATORS, t_generate=600.0, n_signals=12)

    styles = {
        "ACC-0": ("tab:purple", "D", "-."),
        "ACC-0c": ("tab:green", "^", "-"),
        "ACC-tail damp": ("tab:orange", "s", "--"),
    }

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.loglog(TIME_POINTS, analytic, "k-", lw=2.2, label="Analytic")
    for k in ESTIMATORS:
        c, m, ls = styles[k]
        ax.loglog(TIME_POINTS, preds[k], color=c, marker=m, ls=ls, lw=1.8, ms=4.0, label=k)
    ax.set_xlabel("Integration time t (s)")
    ax.set_ylabel(r"$\sigma_{\mathrm{mean}}/\sigma_{\mathrm{signal}}$")
    ax.set_title("ACC variants at B=1")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_09_acc_variants_single_B.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
