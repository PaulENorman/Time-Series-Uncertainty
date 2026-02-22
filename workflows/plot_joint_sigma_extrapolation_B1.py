import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from mocket.noise import generate_bl_noise
from mocket.estimators import bm_var_mean, var_mean_bl, log_var_mean_bl
from mocket.workflows.paths import FIG_DIR

DT = 2e-4
B_TRUE = 1.0
T_GENERATE = 300.0
TIME_POINTS = np.logspace(np.log10(0.2), np.log10(100.0), 24)
N_WINDOWS = 120
SEED = 11

MIN_B = 20
FIT_B = 10
SIGMA2_CLIP_LO = 0.5
SIGMA2_CLIP_HI = 2.0


def _joint_fit_params(window, min_b=MIN_B, fit_b=FIT_B):
    window = np.asarray(window, dtype=float)
    n = window.size
    var_sig = float(np.var(window))
    if n < 20 or var_sig <= 0:
        return np.nan, np.nan, var_sig

    bl_end = int(n / min_b)
    if bl_end < 2:
        return np.nan, np.nan, var_sig

    var_est = bm_var_mean(window, bl_end) / max(var_sig, 1e-15)
    B_est = max(1e-12, 1.0 / (2.0 * max(var_est, 1e-12) * bl_end))
    bl_start = max(2, int(1.0 / (2.0 * B_est)))

    batches = np.logspace(np.log10(bl_start), np.log10(bl_end), fit_b).astype(int)
    batches = np.unique(batches[(batches >= 2) & (batches <= bl_end)])
    if batches.size < 3:
        return B_est, var_sig, var_sig

    varb = np.array([bm_var_mean(window, int(b)) for b in batches], dtype=float)
    valid = np.isfinite(varb) & (varb > 0)
    if valid.sum() < 3:
        return B_est, var_sig, var_sig

    x = batches[valid].astype(float)
    y = np.log(varb[valid])

    def log_abs_var_model(t, log_sigma2, B):
        return log_sigma2 + log_var_mean_bl(t, B)

    log_sigma2_0 = float(np.log(max(var_sig, 1e-15)))
    try:
        popt = curve_fit(
            log_abs_var_model,
            xdata=x,
            ydata=y,
            p0=(log_sigma2_0, B_est),
            bounds=((-40.0, 1e-12), (40.0, np.inf)),
        )[0]
        sigma2_inf = float(np.exp(popt[0]))
        B_fit = float(popt[1])
    except Exception:
        sigma2_inf = var_sig
        B_fit = B_est

    lo = max(1e-15, SIGMA2_CLIP_LO * var_sig)
    hi = max(lo, SIGMA2_CLIP_HI * var_sig)
    sigma2_inf = float(np.clip(sigma2_inf, lo, hi))

    return B_fit, sigma2_inf, var_sig


def main():
    rng = np.random.default_rng(SEED)
    n_long = int(round(T_GENERATE / DT))
    sig = generate_bl_noise(B_TRUE, T_GENERATE, DT, seed=SEED)

    obs_sigma = []
    ext_sigma = []
    obs_sigma_std = []
    ext_sigma_std = []

    for t in TIME_POINTS:
        n_eval = int(round(t / DT))
        max_start = max(1, n_long - n_eval)

        sigmas_window = []
        sigmas_ext = []

        for _ in range(N_WINDOWS):
            start = int(rng.integers(0, max_start))
            w = sig[start:start + n_eval]

            sigmas_window.append(float(np.std(w)))

            _, sigma2_inf, _ = _joint_fit_params(w)
            sigmas_ext.append(float(np.sqrt(sigma2_inf)) if np.isfinite(sigma2_inf) and sigma2_inf > 0 else np.nan)

        arr_w = np.array(sigmas_window, dtype=float)
        arr_e = np.array(sigmas_ext, dtype=float)
        arr_e = arr_e[np.isfinite(arr_e)]

        obs_sigma.append(float(np.mean(arr_w)))
        obs_sigma_std.append(float(np.std(arr_w)))

        ext_sigma.append(float(np.mean(arr_e)) if arr_e.size else np.nan)
        ext_sigma_std.append(float(np.std(arr_e)) if arr_e.size else np.nan)

    obs_sigma = np.array(obs_sigma)
    ext_sigma = np.array(ext_sigma)
    obs_sigma_std = np.array(obs_sigma_std)
    ext_sigma_std = np.array(ext_sigma_std)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.semilogx(TIME_POINTS, obs_sigma, "o-", color="tab:blue", label="Observed window sigma")
    ax.semilogx(TIME_POINTS, ext_sigma, "s--", color="tab:purple", label="BM-joint extrapolated sigma_inf")
    ax.axhline(1.0, color="k", linestyle=":", linewidth=1.5, label="True global sigma = 1")

    ax.fill_between(TIME_POINTS, np.maximum(1e-9, obs_sigma - obs_sigma_std), obs_sigma + obs_sigma_std,
                    color="tab:blue", alpha=0.15)
    ax.fill_between(TIME_POINTS, np.maximum(1e-9, ext_sigma - ext_sigma_std), ext_sigma + ext_sigma_std,
                    color="tab:purple", alpha=0.15)

    ax.set_xlabel("Window length t (s)")
    ax.set_ylabel("Standard deviation")
    ax.set_title(f"B={B_TRUE}: observed sigma(t) vs BM-joint extrapolated sigma_inf")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "joint_sigma_extrapolation_B1.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()

    print(f"Saved {out}")


if __name__ == "__main__":
    main()
