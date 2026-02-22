import numpy as np
import matplotlib.pyplot as plt

from mocket.estimators import (
    var_mean_bl_fit,
)
from mocket.workflows.paths import FIG_DIR


def _ar1_signal(n, rho, seed):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, np.sqrt(1.0 - rho * rho), size=n)
    x = np.zeros(n, dtype=float)
    x[0] = eps[0]
    for k in range(1, n):
        x[k] = rho * x[k - 1] + eps[k]
    return x


def main():
    n = 12000
    dt = 0.01
    rho = 0.97
    x = _ar1_signal(n=n, rho=rho, seed=8)
    t = np.arange(n) * dt

    panel_t_max = 3.5
    panel_n = int(round(panel_t_max / dt)) + 1
    x_panel = x[:panel_n]
    t_panel = t[:panel_n]

    corr_lag1 = float(np.corrcoef(x_panel[:-1], x_panel[1:])[0, 1])

    cum_mean = np.cumsum(x_panel) / np.arange(1, panel_n + 1)
    mean_global = float(np.mean(x_panel))
    std_global = float(np.std(x_panel, ddof=1))

    est_methods = [
        ("i.i.d. sigma/sqrt(n)", (std_global / np.sqrt(panel_n)) ** 2, "k"),
        ("BM", var_mean_bl_fit(x_panel, overlapping=False), "tab:blue"),
    ]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[1.0, 1.4])

    ax0.plot(t_panel, x_panel, color="0.2", linewidth=1.4)
    pts = np.arange(0, panel_n, 8)
    ax0.scatter(t_panel[pts], x_panel[pts], s=14, color="tab:red", zorder=3)
    for i in pts[:18]:
        if i + 1 < panel_n:
            ax0.plot([t_panel[i], t_panel[i + 1]], [x_panel[i], x_panel[i + 1]], color="tab:red", alpha=0.45, linewidth=1.2)
    ax0.set_title(f"Figure 1A — Zoomed trace: adjacent samples are correlated (lag-1 corr = {corr_lag1:.2f})")
    ax0.set_ylabel("Signal")
    ax0.set_xlim(t_panel[0], t_panel[-1])
    ax0.grid(True, alpha=0.25)

    ax1.plot(t_panel, x_panel, color="0.75", linewidth=0.6, label="Signal")
    ax1.plot(t_panel, cum_mean, color="tab:blue", linewidth=2.0, label="Cumulative forward mean")
    ax1.axhline(mean_global, color="k", linestyle="--", linewidth=1.2, label=f"Final mean = {mean_global:+.3f}")

    sigma_vals = []
    for _, var_est, _ in est_methods:
        sigma_vals.append(float(np.sqrt(max(var_est, 0.0))))

    sigma_ref = max(sigma_vals)
    zoom_half = max(4.0 * sigma_ref, 0.03 * std_global)
    y_lo = mean_global - zoom_half
    y_hi = mean_global + zoom_half

    y0_min = float(np.min(x_panel))
    y0_max = float(np.max(x_panel))
    y0_pad = 0.08 * max(1e-12, y0_max - y0_min)
    ax0.set_ylim(y0_min - y0_pad, y0_max + y0_pad)
    x_left = t_panel[0]
    x_right = t_panel[-1]
    x_pad = 0.14 * (x_right - x_left)
    ax1.set_xlim(x_left, x_right + x_pad)
    ax1.set_ylim(y_lo, y_hi)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Signal / mean")
    ax1.set_title("Figure 1B — Cumulative mean with i.i.d. and BM uncertainty brackets")

    # Draw uncertainty brackets outside the right edge of the signal.
    cap = 0.012 * (x_right - x_left)
    x_base = x_right + 0.03 * (x_right - x_left)
    for i, ((label, _, color), sigma_mean) in enumerate(zip(est_methods, sigma_vals)):
        xb = x_base + i * 0.05 * (x_right - x_left)
        y0 = mean_global - sigma_mean
        y1 = mean_global + sigma_mean
        ax1.plot([xb, xb], [y0, y1], color=color, linewidth=2.0)
        ax1.plot([xb - cap, xb], [y0, y0], color=color, linewidth=2.0)
        ax1.plot([xb - cap, xb], [y1, y1], color=color, linewidth=2.0)
        ax1.text(xb + 0.006 * (x_right - x_left), y1, label, color=color, fontsize=8, va="bottom")

    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right", fontsize=8)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_01_effective_sample_size.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
