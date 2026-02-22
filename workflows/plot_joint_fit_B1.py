import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from mocket.noise import generate_bl_noise
from mocket.estimators import bm_var_mean, var_mean_bl, log_var_mean_bl
from mocket.workflows.paths import FIG_DIR

DT = 2e-4
B_TRUE = 1.0
T_WINDOW = 10.0
MIN_B = 20
FIT_B = 12
SEED = 7


def fit_bm_and_joint(window, min_b=MIN_B, fit_b=FIT_B):
    n = window.size
    var_sig = float(np.var(window))
    bl_end = int(n / min_b)

    var_est = bm_var_mean(window, bl_end) / max(var_sig, 1e-15)
    B_est = max(1e-12, 1.0 / (2.0 * max(var_est, 1e-12) * bl_end))
    bl_start = max(2, int(1.0 / (2.0 * B_est)))

    batches = np.logspace(np.log10(bl_start), np.log10(bl_end), fit_b).astype(int)
    batches = np.unique(batches[(batches >= 2) & (batches <= bl_end)])

    varb = np.array([bm_var_mean(window, int(b)) for b in batches], dtype=float)
    valid = np.isfinite(varb) & (varb > 0)
    x = batches[valid].astype(float)
    y_norm = np.log(varb[valid] / max(var_sig, 1e-15))
    y_abs = np.log(varb[valid])

    try:
        B_bm = float(
            curve_fit(
                log_var_mean_bl,
                xdata=x,
                ydata=y_norm,
                p0=B_est,
                bounds=(1e-12, np.inf),
            )[0][0]
        )
    except Exception:
        B_bm = B_est

    def log_abs_model(t, log_sigma2, B):
        return log_sigma2 + log_var_mean_bl(t, B)

    log_sigma2_0 = float(np.log(max(var_sig, 1e-15)))
    try:
        popt = curve_fit(
            log_abs_model,
            xdata=x,
            ydata=y_abs,
            p0=(log_sigma2_0, B_est),
            bounds=((-40.0, 1e-12), (40.0, np.inf)),
        )[0]
        sigma2_inf = float(np.exp(popt[0]))
        B_joint = float(popt[1])
    except Exception:
        sigma2_inf = var_sig
        B_joint = B_est

    sigma2_inf = float(np.clip(sigma2_inf, 0.5 * var_sig, 2.0 * var_sig))

    return x, varb[valid], var_sig, B_bm, B_joint, sigma2_inf


def main():
    n = int(round(T_WINDOW / DT))
    window = generate_bl_noise(B_TRUE, T_WINDOW, DT, seed=SEED)[:n]

    batches, varb_obs, var_sig, B_bm, B_joint, sigma2_inf = fit_bm_and_joint(window)

    obs_ratio = np.sqrt(varb_obs / var_sig)
    bm_ratio = np.sqrt(var_mean_bl(batches, B_bm))
    joint_ratio = np.sqrt((sigma2_inf / var_sig) * var_mean_bl(batches, B_joint))

    obs_abs = np.sqrt(varb_obs)
    bm_abs = np.sqrt(var_sig * var_mean_bl(batches, B_bm))
    joint_abs = np.sqrt(sigma2_inf * var_mean_bl(batches, B_joint))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].loglog(batches, obs_ratio, "ko", label="Observed")
    axs[0].loglog(batches, bm_ratio, "-", color="tab:blue", lw=2, label=f"BM fit  (B={B_bm:.3g})")
    axs[0].loglog(batches, joint_ratio, "--", color="tab:purple", lw=2,
                  label=f"BM-joint fit  (B={B_joint:.3g}, sigma_inf/sigma_win={np.sqrt(sigma2_inf/var_sig):.2f})")
    axs[0].set_xlabel("Batch length b (samples)")
    axs[0].set_ylabel(r"$\sigma_{\mathrm{batch\ mean}} / \sigma_{\mathrm{window}}$")
    axs[0].set_title("Normalized fit view")
    axs[0].grid(True, which="both", alpha=0.3)
    axs[0].legend(fontsize=8)

    axs[1].loglog(batches, obs_abs, "ko", label="Observed")
    axs[1].loglog(batches, bm_abs, "-", color="tab:blue", lw=2, label="BM absolute fit")
    axs[1].loglog(batches, joint_abs, "--", color="tab:purple", lw=2, label="BM-joint absolute fit")
    axs[1].set_xlabel("Batch length b (samples)")
    axs[1].set_ylabel(r"$\sigma_{\mathrm{batch\ mean}}$")
    axs[1].set_title("Absolute fit view")
    axs[1].grid(True, which="both", alpha=0.3)
    axs[1].legend(fontsize=8)

    fig.suptitle(f"B=1.0 diagnostic, T={T_WINDOW}s, dt={DT}, seed={SEED}")
    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "joint_fit_B1.png"
    plt.savefig(out, dpi=170)
    plt.close()

    print(f"Saved {out}")
    print(f"B_bm={B_bm:.6g}, B_joint={B_joint:.6g}, sigma2_window={var_sig:.6g}, sigma2_inf={sigma2_inf:.6g}")


if __name__ == "__main__":
    main()
