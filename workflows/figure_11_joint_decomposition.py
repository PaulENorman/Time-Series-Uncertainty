import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from mocket.estimators import bm_var_mean, log_var_mean_bl, var_mean_bl, var_mean_bl_fit, var_mean_bl_joint_fit
from mocket.noise import generate_bl_noise
from mocket.workflows.paths import FIG_DIR
from mocket.workflows.plot_eval import DT, TIME_POINTS


B_TRUE = 0.03
N_SIGNALS = 20
T_GENERATE = 300.0
SEED = 0
MIN_B = 20
FIT_B = 10


def _fit_b_bm(signal, min_b=MIN_B, fit_b=FIT_B):
    signal = np.asarray(signal, dtype=float)
    n = signal.size
    var_sig = np.var(signal)
    if n < 20 or var_sig <= 0:
        return None

    bl_end = int(n / min_b)
    if bl_end < 2:
        return None

    var_est = bm_var_mean(signal, bl_end) / max(var_sig, 1e-15)
    b0 = max(1e-12, 1.0 / (2.0 * max(var_est, 1e-12) * bl_end))
    bl_start = max(2, int(1.0 / (2.0 * b0)))

    batches = np.logspace(np.log10(bl_start), np.log10(bl_end), fit_b).astype(int)
    batches = np.unique(batches[(batches >= 2) & (batches <= bl_end)])
    if batches.size < 3:
        return None

    varb = np.array([bm_var_mean(signal, int(b)) for b in batches], dtype=float)
    valid = np.isfinite(varb) & (varb > 0)
    if valid.sum() < 3:
        return None

    try:
        b_hat = curve_fit(
            log_var_mean_bl,
            xdata=batches[valid].astype(float),
            ydata=np.log(varb[valid] / var_sig),
            p0=b0,
            bounds=(1e-12, np.inf),
        )[0][0]
    except Exception:
        b_hat = b0
    return float(b_hat)


def _fit_joint_components(signal, min_b=MIN_B, fit_b=FIT_B):
    signal = np.asarray(signal, dtype=float)
    n = signal.size
    var_sig = np.var(signal)
    if n < 20 or var_sig <= 0:
        return None, None, None

    bl_end = int(n / min_b)
    if bl_end < 2:
        return None, None, None

    var_est = bm_var_mean(signal, bl_end) / max(var_sig, 1e-15)
    b0 = max(1e-12, 1.0 / (2.0 * max(var_est, 1e-12) * bl_end))
    bl_start = max(2, int(1.0 / (2.0 * b0)))
    batches = np.logspace(np.log10(bl_start), np.log10(bl_end), fit_b).astype(int)
    batches = np.unique(batches[(batches >= 2) & (batches <= bl_end)])
    if batches.size < 3:
        return None, None, None

    varb = np.array([bm_var_mean(signal, int(b)) for b in batches], dtype=float)
    valid = np.isfinite(varb) & (varb > 0)
    if valid.sum() < 3:
        return None, None, None

    x = batches[valid].astype(float)
    y = np.log(varb[valid])

    def log_abs_var_model(t, log_sigma2, b):
        return log_sigma2 + log_var_mean_bl(t, b)

    log_sigma2_0 = float(np.log(max(var_sig, 1e-15)))
    try:
        popt = curve_fit(
            log_abs_var_model,
            xdata=x,
            ydata=y,
            p0=(log_sigma2_0, b0),
            bounds=((-40.0, 1e-12), (40.0, np.inf)),
        )[0]
        sigma2 = float(np.exp(popt[0]))
        b_hat = float(popt[1])
    except Exception:
        sigma2 = float(var_sig)
        b_hat = float(b0)
    return sigma2, b_hat, float(var_sig)


def compute_decomposition_curves():
    n_long = int(round(T_GENERATE / DT))
    rng = np.random.default_rng(SEED)
    signals = [
        generate_bl_noise(B_TRUE, T_GENERATE, DT, seed=SEED * 10000 + i)
        for i in range(N_SIGNALS)
    ]

    analytic = []
    pred_bm = []
    pred_joint_default = []
    pred_joint_sigma_locked = []
    pred_joint_b_locked = []

    for t in TIME_POINTS:
        n_eval = int(round(t / DT))
        max_start = max(1, n_long - n_eval)
        analytic.append(float(np.sqrt(var_mean_bl(t, B_TRUE))))

        bm_vals = []
        joint_default_vals = []
        sigma_locked_vals = []
        b_locked_vals = []

        for sig in signals:
            start = int(rng.integers(0, max_start))
            w = sig[start: start + n_eval]
            vs = np.var(w)
            if vs <= 0:
                continue

            est_bm = var_mean_bl_fit(w, overlapping=False)
            est_joint_default = var_mean_bl_joint_fit(w)
            sigma2_joint, b_joint, var_sig = _fit_joint_components(w)
            b_bm = _fit_b_bm(w)

            if est_bm > 0:
                bm_vals.append(float(np.sqrt(est_bm / vs)))
            if est_joint_default > 0:
                joint_default_vals.append(float(np.sqrt(est_joint_default / vs)))

            if sigma2_joint is not None and b_joint is not None and var_sig is not None:
                est_sigma_locked = var_sig * var_mean_bl(len(w), b_joint)
                if est_sigma_locked > 0:
                    sigma_locked_vals.append(float(np.sqrt(est_sigma_locked / vs)))

                if b_bm is not None:
                    est_b_locked = sigma2_joint * var_mean_bl(len(w), b_bm)
                    if est_b_locked > 0:
                        b_locked_vals.append(float(np.sqrt(est_b_locked / vs)))

        pred_bm.append(float(np.nanmean(bm_vals)) if bm_vals else np.nan)
        pred_joint_default.append(float(np.nanmean(joint_default_vals)) if joint_default_vals else np.nan)
        pred_joint_sigma_locked.append(float(np.nanmean(sigma_locked_vals)) if sigma_locked_vals else np.nan)
        pred_joint_b_locked.append(float(np.nanmean(b_locked_vals)) if b_locked_vals else np.nan)

    return (
        np.array(analytic),
        np.array(pred_bm),
        np.array(pred_joint_default),
        np.array(pred_joint_sigma_locked),
        np.array(pred_joint_b_locked),
    )


def main():
    analytic, pred_bm, pred_joint_default, pred_joint_sigma_locked, pred_joint_b_locked = compute_decomposition_curves()

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.loglog(TIME_POINTS, analytic, "k-", linewidth=2.5, label="Analytic target")
    ax.plot(TIME_POINTS, pred_bm, color="tab:blue", marker="o", linestyle="-", linewidth=1.8, markersize=4, alpha=0.9, label="BM")
    ax.plot(TIME_POINTS, pred_joint_default, color="tab:purple", marker="*", linestyle="--", linewidth=2.0, markersize=5, alpha=0.9, label="BM-joint (default)")
    ax.plot(TIME_POINTS, pred_joint_sigma_locked, color="tab:green", marker="s", linestyle="-.", linewidth=1.9, markersize=4, alpha=0.9, label="Joint B-only (sigma locked)")
    ax.plot(TIME_POINTS, pred_joint_b_locked, color="tab:orange", marker="D", linestyle=":", linewidth=2.0, markersize=4, alpha=0.9, label="Joint sigma-only (B locked to BM)")

    ax.set_xlabel("Integration time  t  (s)")
    ax.set_ylabel(r"$\sigma_\mathrm{mean}\,/\,\sigma_\mathrm{signal}$")
    ax.set_title("Figure 11 — BM-joint decomposition (B=0.03)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_11_joint_decomposition.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
