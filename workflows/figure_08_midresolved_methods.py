import matplotlib.pyplot as plt
import numpy as np
from mocket.noise import generate_bl_noise
from mocket.estimators import var_mean_bl, var_mean_bl_fit, var_mean_bl_joint_fit
from mocket.workflows.plot_eval import TIME_POINTS, DT
from mocket.workflows.paths import FIG_DIR


B_TRUE = 0.03
N_SIGNALS = 20
T_GENERATE = 300.0
SEED = 0


def compute_clipping_curves():
    n_long = int(round(T_GENERATE / DT))
    rng = np.random.default_rng(SEED)

    signals = [
        generate_bl_noise(B_TRUE, T_GENERATE, DT, seed=SEED * 10000 + i)
        for i in range(N_SIGNALS)
    ]

    analytic = []
    pred_bm = []
    pred_joint_default = []

    for t in TIME_POINTS:
        n_eval = int(round(t / DT))
        max_start = max(1, n_long - n_eval)

        analytic.append(float(np.sqrt(var_mean_bl(t, B_TRUE))))

        bm_vals = []
        joint_default_vals = []

        for sig in signals:
            start = int(rng.integers(0, max_start))
            w = sig[start: start + n_eval]
            vs = np.var(w)
            if vs <= 0:
                continue

            est_bm = var_mean_bl_fit(w, overlapping=False)
            est_joint_default = var_mean_bl_joint_fit(w)

            if est_bm > 0:
                bm_vals.append(float(np.sqrt(est_bm / vs)))
            if est_joint_default > 0:
                joint_default_vals.append(float(np.sqrt(est_joint_default / vs)))

        pred_bm.append(float(np.nanmean(bm_vals)) if bm_vals else np.nan)
        pred_joint_default.append(float(np.nanmean(joint_default_vals)) if joint_default_vals else np.nan)

    return (
        np.array(analytic),
        np.array(pred_bm),
        np.array(pred_joint_default),
    )


def main():
    analytic, pred_bm, pred_joint_default = compute_clipping_curves()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(TIME_POINTS, analytic, "k-", linewidth=2.5, label="Analytic target")

    ax.plot(
        TIME_POINTS,
        pred_bm,
        color="tab:blue",
        marker="o",
        linestyle="-",
        linewidth=1.8,
        markersize=4,
        alpha=0.9,
        label="BM",
    )
    ax.plot(
        TIME_POINTS,
        pred_joint_default,
        color="tab:purple",
        marker="*",
        linestyle="--",
        linewidth=2.0,
        markersize=5,
        alpha=0.9,
        label="BM-joint (default)",
    )

    ax.set_xlabel("Integration time  t  (s)")
    ax.set_ylabel(r"$\sigma_\mathrm{mean}\,/\,\sigma_\mathrm{signal}$")
    ax.set_title("Figure 8 — Mid-resolved: BM vs BM-joint default (B=0.03)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_08_midresolved_methods.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
