import matplotlib.pyplot as plt

from mocket.workflows.paths import FIG_DIR
from mocket.workflows.plot_eval import TIME_POINTS, compute_curves


# (title, bandwidth, regime label)
CASES = [
    ("B = 0.005", 0.005, "underresolved"),
    ("B = 0.012", 0.012, "near limit"),
    ("B = 0.10", 0.10, "resolved"),
    ("B = 15", 15.0, "overresolved"),
]


def main():
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.2), sharex=True, sharey=True)
    axs = axs.ravel()

    for ax, (name, b, regime) in zip(axs, CASES):
        analytic, preds = compute_curves(b)
        ax.loglog(TIME_POINTS, analytic, "k-", linewidth=2.1, label="Analytic")
        ax.loglog(TIME_POINTS, preds["BM"], color="tab:blue", marker="o", markersize=3.6, linewidth=1.6, label="BM")
        ax.set_title(f"{name} ({regime})", fontsize=10)
        ax.grid(True, which="both", alpha=0.25)

    axs[0].set_ylabel(r"$\sigma_\mathrm{mean}/\sigma_\mathrm{signal}$")
    axs[2].set_ylabel(r"$\sigma_\mathrm{mean}/\sigma_\mathrm{signal}$")
    axs[2].set_xlabel("Integration time t (s)")
    axs[3].set_xlabel("Integration time t (s)")
    axs[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Figure 14 — BM behavior across under/near/resolved/overresolved regimes", y=0.995)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_14_bm_regime_overview.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
