import numpy as np
import matplotlib.pyplot as plt

from mocket.estimators import var_mean_bl, var_mean_bl_fit
from mocket.noise import generate_bl_noise
from mocket.workflows.paths import FIG_DIR
from mocket.workflows.plot_eval import DT, N_SIGNALS, T_GENERATE, TIME_POINTS


CASES = [
    ("B = 0.005", 0.005, "underresolved"),
    ("B = 0.012", 0.012, "near limit"),
    ("B = 0.10", 0.10, "resolved"),
    ("B = 15", 15.0, "overresolved"),
]


def _compute_curves(B, estimators, seed=0):
    n_long = int(round(T_GENERATE / DT))
    rng = np.random.default_rng(seed)

    signals = [
        generate_bl_noise(B, T_GENERATE, DT, seed=seed * 10000 + i)
        for i in range(N_SIGNALS)
    ]

    analytic_curve = []
    pred_curves = {label: [] for label in estimators}

    for t in TIME_POINTS:
        n_eval = int(round(t / DT))
        max_start = max(1, n_long - n_eval)
        analytic_curve.append(float(np.sqrt(var_mean_bl(t, B))))

        for label, fn in estimators.items():
            preds = []
            for sig in signals:
                start = int(rng.integers(0, max_start))
                w = sig[start: start + n_eval]
                vs = np.var(w)
                if vs <= 0:
                    continue
                est = fn(w)
                if est > 0:
                    preds.append(float(np.sqrt(est / vs)))
            pred_curves[label].append(float(np.nanmean(preds)) if preds else np.nan)

    return np.array(analytic_curve), {k: np.array(v) for k, v in pred_curves.items()}


def main():
    estimators = {
        "BM": lambda w: var_mean_bl_fit(w, overlapping=False),
        "OBM 1%": lambda w: var_mean_bl_fit(w, overlapping=True, overlap_ratio=0.01),
        "OBM 10%": lambda w: var_mean_bl_fit(w, overlapping=True, overlap_ratio=0.1),
        "OBM 25%": lambda w: var_mean_bl_fit(w, overlapping=True, overlap_ratio=0.25),
        "OBM 50%": lambda w: var_mean_bl_fit(w, overlapping=True, overlap_ratio=0.5),
        "OBM 75%": lambda w: var_mean_bl_fit(w, overlapping=True, overlap_ratio=0.75),
    }

    fig, axs = plt.subplots(2, 2, figsize=(11.2, 8.4), sharex=True, sharey=True)
    axs = axs.ravel()

    styles = {
        "BM": ("0.35", "o", "-"),
        "OBM 1%": ("tab:red", "v", "-"),
        "OBM 10%": ("tab:blue", ">", "-."),
        "OBM 25%": ("tab:green", "^", "-."),
        "OBM 50%": ("tab:orange", "s", "--"),
        "OBM 75%": ("tab:purple", "D", ":"),
    }

    for i, (ax, (name, b, regime)) in enumerate(zip(axs, CASES)):
        analytic, preds = _compute_curves(b, estimators, seed=100 + i)
        ax.loglog(TIME_POINTS, analytic, "k-", linewidth=2.0, label="Analytic")
        for label in estimators:
            c, m, ls = styles[label]
            ax.loglog(TIME_POINTS, preds[label], color=c, marker=m, linestyle=ls, linewidth=1.6, markersize=3.4, label=label)
        ax.set_title(f"{name} ({regime})", fontsize=10)
        ax.grid(True, which="both", alpha=0.25)

    axs[0].set_ylabel(r"$\sigma_\mathrm{mean}/\sigma_\mathrm{signal}$")
    axs[2].set_ylabel(r"$\sigma_\mathrm{mean}/\sigma_\mathrm{signal}$")
    axs[2].set_xlabel("Integration time t (s)")
    axs[3].set_xlabel("Integration time t (s)")
    axs[0].legend(fontsize=7.2, loc="upper right")
    fig.suptitle("Figure 16 — OBM overlap-ratio effect across regimes", y=0.995)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_16_obm_overlap_regime_overview.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
