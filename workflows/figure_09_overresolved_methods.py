import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from mocket.noise import generate_bl_noise
from mocket.estimators import bm_var_mean, var_mean_bl, var_bl
from mocket.workflows.paths import FIG_DIR


DT = 2e-4
B_TRUE = 15.0
T_WINDOW = 2.5
MIN_B = 14
FIT_B = 16
SEED = 72


def main():
    n = int(round(T_WINDOW / DT))
    signal = generate_bl_noise(B_TRUE, T_WINDOW, DT, seed=SEED)[:n]
    sigma2_win = float(np.var(signal))
    sigma_win = float(np.sqrt(max(sigma2_win, 1e-15)))

    bl_end = int(n / MIN_B)
    batches = np.unique(np.logspace(np.log10(20), np.log10(max(25, bl_end)), FIT_B).astype(int))

    bm_obs = []
    wv_obs = []
    valid_b = []

    for b in batches:
        b = int(b)
        nb = n // b
        if nb < 2:
            continue
        v_bm = float(bm_var_mean(signal, b))
        segs = signal[: nb * b].reshape(nb, b)
        v_within = float(np.mean(np.var(segs, axis=1, ddof=1)))
        if np.isfinite(v_bm) and np.isfinite(v_within) and v_bm > 0 and v_within > 0:
            bm_obs.append(v_bm)
            wv_obs.append(v_within)
            valid_b.append(float(b))

    b = np.asarray(valid_b, dtype=float)
    y_bm = np.log(np.asarray(bm_obs, dtype=float))
    y_wv = np.log(np.asarray(wv_obs, dtype=float))

    var_est = bm_var_mean(signal, max(2, bl_end)) / max(sigma2_win, 1e-15)
    B0 = max(1e-12, 1.0 / (2.0 * max(var_est, 1e-12) * max(2, bl_end)))
    miss_frac0 = float(np.clip(var_mean_bl(n, B0), 1e-6, 0.999))
    sigma2_0 = max(sigma2_win / (1.0 - miss_frac0), sigma2_win, 1e-15)

    x_t = np.concatenate([b, b])
    x_kind = np.concatenate([np.zeros_like(b), np.ones_like(b)])
    y_all = np.concatenate([y_bm, y_wv])

    def log_dual_model(xdata, log_sigma2, B):
        t = xdata[0]
        kind = xdata[1]
        pred_bm = log_sigma2 + np.log(np.clip(var_mean_bl(t, B), 1e-15, None))
        pred_wv = log_sigma2 + np.log(np.clip(t / (t - 1.0), 1e-15, None)) + np.log(np.clip(var_bl(t, B), 1e-15, None))
        return np.where(kind < 0.5, pred_bm, pred_wv)

    try:
        popt = curve_fit(
            log_dual_model,
            xdata=np.vstack([x_t, x_kind]),
            ydata=y_all,
            p0=(np.log(sigma2_0), B0),
            bounds=((-40.0, 1e-12), (40.0, np.inf)),
        )[0]
        sigma2_dual = float(np.exp(popt[0]))
        B_dual = float(popt[1])
    except Exception:
        sigma2_dual = sigma2_win
        B_dual = B0

    sigma2_dual = float(np.clip(sigma2_dual, 0.5 * sigma2_win, 2.0 * sigma2_win))

    b_dense = np.logspace(np.log10(np.min(b)), np.log10(np.max(b)), 260)
    bm_std_obs = np.sqrt(np.asarray(bm_obs, dtype=float)) / sigma_win
    wv_std_obs = np.sqrt(np.asarray(wv_obs, dtype=float)) / sigma_win

    bm_fit = np.sqrt(np.clip(sigma2_dual * var_mean_bl(b_dense, B_dual), 1e-15, None)) / sigma_win
    wv_fit = np.sqrt(np.clip(sigma2_dual * (b_dense / (b_dense - 1.0)) * var_bl(b_dense, B_dual), 1e-15, None)) / sigma_win

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.loglog(b * DT, bm_std_obs, "o", color="tab:blue", ms=4.5, label="Observed batch-mean std / sigma_win")
    ax.loglog(b * DT, wv_std_obs, "s", color="tab:green", ms=4.5, label="Observed within-batch std / sigma_win")
    ax.loglog(b_dense * DT, bm_fit, "-", color="tab:blue", lw=2.0, label="Dual fit (batch-mean channel)")
    ax.loglog(b_dense * DT, wv_fit, "-", color="tab:green", lw=2.0, label=f"Dual fit (within-batch channel), B={B_dual:.3g} Hz")

    ax.set_xlabel("Batch length b (seconds)")
    ax.set_ylabel("Normalized std")
    ax.set_title("Figure 9 — Dual method: fitting B using both observables")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_09_overresolved_methods.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
