import numpy as np
import matplotlib.pyplot as plt

from mocket.noise import generate_bl_noise
from mocket.estimators import var_mean_bl
from mocket.workflows.paths import FIG_DIR


DT = 2e-4
B = 15.0
T_GENERATE = 300.0
N_SIGNALS = 180
TIME_POINTS = np.logspace(np.log10(0.4), np.log10(90.0), 24)
SEED = 31


def main():
    n_long = int(round(T_GENERATE / DT))
    rng = np.random.default_rng(SEED)

    signals = [
        generate_bl_noise(B, T_GENERATE, DT, seed=SEED * 10000 + i)
        for i in range(N_SIGNALS)
    ]

    empirical = []
    for t in TIME_POINTS:
        n_eval = int(round(t / DT))
        max_start = max(1, n_long - n_eval)
        means = []
        stds = []
        for sig in signals:
            start = int(rng.integers(0, max_start))
            w = sig[start:start + n_eval]
            means.append(float(np.mean(w)))
            stds.append(float(np.std(w)))
        empirical.append(float(np.std(means) / np.mean(stds)))

    empirical = np.asarray(empirical)
    analytic = np.sqrt(var_mean_bl(TIME_POINTS, B))

    t_tail = np.logspace(np.log10(4.0), np.log10(90.0), 100)
    c = empirical[-1] * np.sqrt(TIME_POINTS[-1])
    power_guide = c * t_tail ** (-0.5)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.loglog(TIME_POINTS, empirical, "o-", color="tab:blue", lw=1.6, ms=4, label="Empirical (Monte Carlo)")
    ax.loglog(TIME_POINTS, analytic, "k-", lw=2.2, label="Analytic BL")
    ax.loglog(t_tail, power_guide, "--", color="tab:red", lw=1.8, label=r"Tail guide: $t^{-1/2}$")

    ax.set_xlabel("Integration time t (s)")
    ax.set_ylabel(r"$\sigma_{\mathrm{mean}}/\sigma_{\mathrm{signal}}$")
    ax.set_title(f"Figure 6 — Long-time scaling validation with real data points (B={B:g} Hz)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_06_long_time_scaling_validation.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
