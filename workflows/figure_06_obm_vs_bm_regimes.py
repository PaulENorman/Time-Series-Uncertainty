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
from plot_eval import TIME_POINTS, compute_curves


CASES = [("B=0.1", 0.1), ("B=1", 1.0), ("B=10", 10.0)]


def main():
    fig, axs = plt.subplots(3, 1, figsize=(8.2, 11.2), sharex=True, sharey=True)
    for ax, (name, b) in zip(axs, CASES):
        analytic, preds = compute_curves(b, estimators=OBM_SWEEP_METHODS, t_generate=600.0, n_signals=12)
        ax.loglog(TIME_POINTS, analytic, "k-", lw=2.0, label="Analytic")
        for k in OBM_SWEEP_METHODS:
            c, m, ls, _ = OBM_SWEEP_STYLE[k]
            ax.loglog(TIME_POINTS, preds[k], color=c, marker=m, ls=ls, lw=1.7, ms=3.8, label=k)
        ax.set_title(name)
        ax.grid(True, which="both", alpha=0.25)
    axs[0].legend(fontsize=8)
    axs[-1].set_xlabel("Integration time t (s)")
    for ax in axs:
        ax.set_ylabel(r"$\sigma_{\mathrm{mean}}/\sigma_{\mathrm{signal}}$")
    fig.suptitle("OBM overlap sweep vs BM across regimes")
    out = FIG_DIR / "figure_06_obm_vs_bm_regimes.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
