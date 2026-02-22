import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from mocket.estimators import bm_var_mean, log_var_mean_bl, var_mean_bl, var_mean_bl_fit
from mocket.noise import generate_bl_noise
from mocket.workflows.paths import FIG_DIR


DT = 2e-4
T_GENERATE = 300.0
N_SIGNALS = 12
TIME_POINTS = np.logspace(np.log10(0.3), np.log10(100.0), 14)
MIN_B = 20
FIT_B = 10
SEED = 55

CASES = [
    ("B=0.005 (underresolved)", 0.005),
    ("B=0.03 (near/resolved)", 0.03),
    ("B=1.0 (well resolved)", 1.0),
]


def _batch_curve(signal):
    n = signal.size
    var_sig = float(np.var(signal))
    if n < 20 or var_sig <= 0:
        return None
    bl_end = int(n / MIN_B)
    if bl_end < 2:
        return None

    var_est = bm_var_mean(signal, bl_end) / max(var_sig, 1e-15)
    b0 = max(1e-12, 1.0 / (2.0 * max(var_est, 1e-12) * bl_end))
    bl_start = max(2, int(1.0 / (2.0 * b0)))
    batches = np.logspace(np.log10(bl_start), np.log10(bl_end), FIT_B).astype(int)
    batches = np.unique(batches[(batches >= 2) & (batches <= bl_end)])
    if batches.size < 4:
        return None

    varb = np.array([bm_var_mean(signal, int(b)) for b in batches], dtype=float)
    valid = np.isfinite(varb) & (varb > 0)
    if valid.sum() < 4:
        return None

    b = batches[valid].astype(float)
    y = np.sqrt(varb[valid] / max(var_sig, 1e-15))
    return b, y, var_sig, b0


def _est_tail_power(signal):
    out = _batch_curve(signal)
    if out is None:
        return np.var(signal)
    b, y, var_sig, _ = out
    t_sec = b * DT
    k = max(4, int(np.ceil(0.35 * len(t_sec))))
    xt = np.log(t_sec[-k:])
    yt = np.log(y[-k:])
    slope, intercept = np.polyfit(xt, yt, 1)
    alpha = float(-slope)
    c = float(np.exp(intercept))
    sigma_ratio = float(c * (len(signal) * DT) ** (-alpha))
    return (sigma_ratio * sigma_ratio) * var_sig


def _est_conservative(signal):
    out = _batch_curve(signal)
    if out is None:
        return np.var(signal)
    b, y, var_sig, _ = out
    t_sec = b * DT
    k = max(4, int(np.ceil(0.35 * len(t_sec))))
    c_cons = float(np.max(y[-k:] * np.sqrt(t_sec[-k:])))
    sigma_ratio = float(c_cons / np.sqrt(len(signal) * DT))
    return (sigma_ratio * sigma_ratio) * var_sig


def _compute_method_curves(B):
    n_long = int(round(T_GENERATE / DT))
    rng = np.random.default_rng(SEED + int(1000 * B))
    signals = [generate_bl_noise(B, T_GENERATE, DT, seed=SEED * 10000 + i) for i in range(N_SIGNALS)]

    analytic = []
    pred_full = []
    pred_tail = []
    pred_cons = []

    for t in TIME_POINTS:
        n_eval = int(round(t / DT))
        max_start = max(1, n_long - n_eval)
        analytic.append(float(np.sqrt(var_mean_bl(t, B))))

        vals_full = []
        vals_tail = []
        vals_cons = []
        for sig in signals:
            start = int(rng.integers(0, max_start))
            w = sig[start:start + n_eval]
            vs = np.var(w)
            if vs <= 0:
                continue

            est_full = var_mean_bl_fit(w, overlapping=False)
            est_tail = _est_tail_power(w)
            est_cons = _est_conservative(w)

            if est_full > 0:
                vals_full.append(float(np.sqrt(est_full / vs)))
            if est_tail > 0:
                vals_tail.append(float(np.sqrt(est_tail / vs)))
            if est_cons > 0:
                vals_cons.append(float(np.sqrt(est_cons / vs)))

        pred_full.append(float(np.nanmean(vals_full)) if vals_full else np.nan)
        pred_tail.append(float(np.nanmean(vals_tail)) if vals_tail else np.nan)
        pred_cons.append(float(np.nanmean(vals_cons)) if vals_cons else np.nan)

    return np.array(analytic), np.array(pred_full), np.array(pred_tail), np.array(pred_cons)


def main():
    fig, axs = plt.subplots(1, 3, figsize=(14.4, 4.7), sharex=True, sharey=True)

    for ax, (title, b) in zip(axs, CASES):
        analytic, y_full, y_tail, y_cons = _compute_method_curves(b)
        ax.loglog(TIME_POINTS, analytic, "k-", lw=2.2, label="Analytic")
        ax.loglog(TIME_POINTS, y_full, color="tab:blue", marker="o", ms=3.8, lw=1.6, label="Full form")
        ax.loglog(TIME_POINTS, y_tail, color="tab:green", marker="^", ms=3.8, lw=1.6, ls="--", label="Tail power")
        ax.loglog(TIME_POINTS, y_cons, color="tab:red", marker="s", ms=3.8, lw=1.6, ls=":", label="Conservative")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Integration time t (s)")
        ax.grid(True, which="both", alpha=0.25)

    axs[0].set_ylabel(r"$\sigma_\mathrm{mean}/\sigma_\mathrm{signal}$")
    axs[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Figure 8 — Expected fit behavior across B regimes", y=0.99)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_08_fitting_regimes.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
