import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import FuncFormatter

from mocket.estimators import bm_var_mean, log_var_mean_bl, var_mean_bl
from mocket.noise import generate_bl_noise
from mocket.workflows.paths import FIG_DIR


DT = 2e-4
B_TRUE = 1.0
T_WINDOW = 40.0
SEED = 21
B_MIN = 60
B_MAX_DIV = 8
FIT_B = 22


def _batch_curve(signal):
    n = signal.size
    var_sig = float(np.var(signal))
    b_min = max(2, B_MIN)
    b_max = max(b_min + 4, int(n / B_MAX_DIV))
    batches = np.logspace(np.log10(b_min), np.log10(b_max), FIT_B).astype(int)
    batches = np.unique(batches[(batches >= 2) & (batches <= b_max)])
    varb = np.array([bm_var_mean(signal, int(b)) for b in batches], dtype=float)
    valid = np.isfinite(varb) & (varb > 0)
    b = batches[valid].astype(float)
    y = np.sqrt(varb[valid] / max(var_sig, 1e-15))
    # Keep B initialization compatible with existing full-form fit.
    bl_end = max(2, int(n / 20))
    var_est = bm_var_mean(signal, bl_end) / max(var_sig, 1e-15)
    b0 = max(1e-12, 1.0 / (2.0 * max(var_est, 1e-12) * bl_end))
    return b, y, var_sig, b0


def _fit_full_form(b, y2, b0):
    try:
        B_hat = float(
            curve_fit(
                log_var_mean_bl,
                xdata=b,
                ydata=np.log(y2),
                p0=b0,
                bounds=(1e-12, np.inf),
            )[0][0]
        )
    except Exception:
        B_hat = float(b0)
    return B_hat


def _fit_tail_power(t_sec, y):
    k = max(4, int(np.ceil(0.35 * len(t_sec))))
    xt = np.log(t_sec[-k:])
    yt = np.log(y[-k:])
    slope, intercept = np.polyfit(xt, yt, 1)
    alpha = float(-slope)
    c = float(np.exp(intercept))
    return alpha, c


def _fit_conservative_envelope(t_sec, y):
    k = max(4, int(np.ceil(0.35 * len(t_sec))))
    c_cons = float(np.max(y[-k:] * np.sqrt(t_sec[-k:])))
    return c_cons


def main():
    n = int(round(T_WINDOW / DT))
    x = generate_bl_noise(B_TRUE, T_WINDOW, DT, seed=SEED)[:n]
    b, y_obs, var_sig, b0 = _batch_curve(x)
    t_sec = b * DT

    # (1) full functional-form fit
    B_hat = _fit_full_form(b, y_obs * y_obs, b0)
    y_full = np.sqrt(var_mean_bl(b, B_hat))

    # (2) tail power-law fit
    alpha, c = _fit_tail_power(t_sec, y_obs)
    y_tail = c * t_sec ** (-alpha)

    # (3) conservative envelope with fixed -1/2 slope
    c_cons = _fit_conservative_envelope(t_sec, y_obs)
    y_cons = c_cons / np.sqrt(t_sec)

    t_end = n * DT
    sigma_full = float(np.sqrt(var_mean_bl(n, B_hat)))
    sigma_tail = float(c * t_end ** (-alpha))
    sigma_cons = float(c_cons / np.sqrt(t_end))

    fig, axs = plt.subplots(1, 3, figsize=(14.6, 4.9), sharex=True, sharey=True)

    panels = [
        ("Full functional-form fit", y_full, f"\\hat B={B_hat:.3g},  σ_mean≈{sigma_full:.3f}σ"),
        ("Tail power-law fit", y_tail, f"α={alpha:.2f},  σ_mean≈{sigma_tail:.3f}σ"),
        ("Conservative envelope fit", y_cons, f"fixed α=0.5,  σ_mean≈{sigma_cons:.3f}σ"),
    ]

    for ax, (title, y_fit, subtitle) in zip(axs, panels):
        ax.loglog(t_sec, y_obs, "ko", ms=4, label="Observed batch-mean std")
        ax.loglog(t_sec, y_fit, color="tab:blue", lw=2.0, label="Fit")
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("Batch length b (s)")
        ax.tick_params(axis="x", labelrotation=28, labelsize=8)

    # Expand y-range so differences are visually clearer.
    y_min = max(1e-4, float(np.nanmin(y_obs) * 0.65))
    y_max = float(np.nanmax(y_obs) * 1.6)
    axs[0].set_ylim(y_min, y_max)

    # Use readable x tick labels across all panels.
    x_ticks = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5], dtype=float)
    x_ticks = x_ticks[(x_ticks >= np.min(t_sec)) & (x_ticks <= np.max(t_sec))]
    for ax in axs:
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    axs[0].set_ylabel(r"$\sigma_{\mathrm{batch\ mean}}/\sigma_{\mathrm{window}}$")
    axs[0].legend(fontsize=8)
    fig.suptitle("Figure 7 — Three fitting approaches on the same batch-means curve", y=0.99)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_07_fitting_approaches.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
