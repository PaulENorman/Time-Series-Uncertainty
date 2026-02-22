import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from mocket.noise import generate_bl_noise
from mocket.estimators import var_bl
from mocket.workflows.paths import FIG_DIR


DT = 2e-4
B_TRUE = 15.0
T_WINDOW = 2.5
FIT_B = 16
SEED = 70


def main():
    n = int(round(T_WINDOW / DT))
    signal = generate_bl_noise(B_TRUE, T_WINDOW, DT, seed=SEED)[:n]
    sigma_win = float(np.std(signal, ddof=1))

    batches = np.unique(np.logspace(np.log10(20), np.log10(max(25, n // 8)), FIT_B).astype(int))

    b_valid = []
    y_std = []
    y_var_norm = []
    for b in batches:
        nb = n // int(b)
        if nb < 2:
            continue
        segs = signal[: nb * int(b)].reshape(nb, int(b))
        within_var = float(np.mean(np.var(segs, axis=1, ddof=1)))
        if np.isfinite(within_var) and within_var > 0:
            b_valid.append(float(b))
            y_std.append(float(np.sqrt(within_var) / max(sigma_win, 1e-15)))
            y_var_norm.append(float(within_var / max(sigma_win**2, 1e-15)))

    b_valid = np.asarray(b_valid, dtype=float)
    y_std = np.asarray(y_std, dtype=float)
    y_var_norm = np.asarray(y_var_norm, dtype=float)

    def log_within_var_model(t, B):
        return np.log(np.clip(t / (t - 1.0), 1e-15, None)) + np.log(np.clip(var_bl(t, B), 1e-15, None))

    B0 = 5.0
    try:
        B_fit = float(
            curve_fit(
                log_within_var_model,
                xdata=b_valid,
                ydata=np.log(np.clip(y_var_norm, 1e-15, None)),
                p0=B0,
                bounds=(1e-12, np.inf),
            )[0][0]
        )
    except Exception:
        B_fit = B0

    b_dense = np.logspace(np.log10(np.min(b_valid)), np.log10(np.max(b_valid)), 240)
    fit_std = np.sqrt(np.clip((b_dense / (b_dense - 1.0)) * var_bl(b_dense, B_fit), 1e-15, None))

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.loglog(b_valid * DT, y_std, "o", color="tab:green", ms=5, label="Observed within-batch std / sigma_win")
    ax.loglog(b_dense * DT, fit_std, "-", color="tab:green", lw=2.0, label=f"BM-std fit (B={B_fit:.3g} Hz)")
    ax.set_xlabel("Batch length b (seconds)")
    ax.set_ylabel(r"$\sigma_{\mathrm{within\ batch}}/\sigma_{\mathrm{window}}$")
    ax.set_title("Figure 7 — BM-std method: fitting within-batch spread")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_07_underresolved_methods.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
