import numpy as np
import matplotlib.pyplot as plt

from mocket.estimators import var_mean_bl, var_mean_bl_fit
from mocket.noise import generate_bl_noise
from mocket.workflows.paths import FIG_DIR
from mocket.workflows.plot_eval import DT, N_SIGNALS, T_GENERATE, TIME_POINTS


B_VALUES = [0.005, 0.008, 0.012, 0.03, 0.1, 1.0, 15.0, 40.0]

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

    maes = {label: [] for label in estimators}

    for b in B_VALUES:
        analytic, preds = _compute_curves(b, estimators)
        valid = np.isfinite(analytic) & (analytic > 0)
        a = analytic[valid]
        for label in estimators:
            p = preds[label][valid]
            mae = float(np.nanmean(np.abs((p - a) / a)))
            maes[label].append(mae)

    x = np.array(B_VALUES, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.semilogx(x, maes["BM"], color="0.35", marker="o", linewidth=1.8, markersize=5, label="BM")
    ax.semilogx(x, maes["OBM 1%"], color="tab:red", marker="v", linewidth=1.8, markersize=5, label="OBM 1%")
    ax.semilogx(x, maes["OBM 10%"], color="tab:blue", marker=">", linewidth=1.8, markersize=5, label="OBM 10%")
    ax.semilogx(x, maes["OBM 25%"], color="tab:green", marker="^", linewidth=1.8, markersize=5, label="OBM 25%")
    ax.semilogx(x, maes["OBM 50%"], color="tab:orange", marker="s", linewidth=1.8, markersize=5, label="OBM 50%")
    ax.semilogx(x, maes["OBM 75%"], color="tab:purple", marker="D", linewidth=1.8, markersize=5, label="OBM 75%")

    ax.set_xlabel("Bandwidth B (Hz)")
    ax.set_ylabel("Mean absolute relative error vs analytic")
    ax.set_title("Figure 15 — OBM performance vs B for overlap ratios")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_15_obm_overlap_performance_vs_B.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
