"""
Plot sigma_mean/sigma_signal vs integration time for a multi-band example,
comparing the shared MOCKET estimator set to both the empirical Monte Carlo
target and the exact analytic mixture formula.

The analytic target for a sum of K independent unit-variance BL components
(normalised to unit variance overall) is:

    sqrt( mean_k( var_mean_bl(t, B_k) ) )

Usage:
    python workflows/plot_multiband.py
"""

import numpy as np
import matplotlib.pyplot as plt

from mocket.estimators import (
    var_mean_bl,
)
from mocket.workflows.method_registry import METHODS, METHOD_STYLE
from mocket.workflows.paths import FIG_DIR
from mocket.noise import generate_multiband_noise

# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #
BANDWIDTHS  = [0.1, 1.0]          # the multi-band example
N_SIGNALS   = 500
DT          = 2e-4
T_GENERATE  = 300.0
TIME_POINTS = np.logspace(np.log10(0.2), np.log10(100.0), 30)
SEED        = 42

# ------------------------------------------------------------------ #
#  Analytic mixture target
# ------------------------------------------------------------------ #
def analytic_mixture(t_arr, bandwidths):
    """sqrt of the mean var_mean_bl across components."""
    vm = np.mean([var_mean_bl(t_arr, B) for B in bandwidths], axis=0)
    return np.sqrt(vm)

# ------------------------------------------------------------------ #
#  Monte Carlo
# ------------------------------------------------------------------ #
def compute(bandwidths, time_points, n_signals, t_generate, dt, seed):
    n_long = int(round(t_generate / dt))
    rng = np.random.default_rng(seed)

    print(f"  Generating {n_signals} signals ...", flush=True)
    signals = [
        generate_multiband_noise(bandwidths, t_generate, dt,
                                 seed=seed * 100_000 + i)
        for i in range(n_signals)
    ]


    empirical_sigma_mean = []
    empirical_sigma_window = []
    pred_sigma_means = {lbl: [] for lbl, _ in METHODS}

    print(f"  Evaluating {len(time_points)} time points ...", flush=True)
    for t in time_points:
        n_eval    = int(round(t / dt))
        max_start = max(1, n_long - n_eval)

        trial_means = []
        trial_window_stds = []
        method_preds = {lbl: [] for lbl, _ in METHODS}

        for sig in signals:
            start = int(rng.integers(0, max_start))
            w = sig[start: start + n_eval]
            trial_means.append(float(np.mean(w)))
            trial_window_stds.append(float(np.std(w, ddof=1)))
            
            vs = np.var(w, ddof=1)
            if vs > 0:
                for lbl, fn in METHODS:
                    try:
                        est = fn(w)
                        if est > 0:
                            # Store raw estimator output (sigma_mean), no normalization
                            method_preds[lbl].append(np.sqrt(est))
                    except Exception:
                        pass

        # Calculate metrics
        empirical_sigma_mean.append(float(np.std(trial_means)))
        empirical_sigma_window.append(float(np.mean(trial_window_stds)))
        
        for lbl, _ in METHODS:
            p = method_preds[lbl]
            pred_sigma_means[lbl].append(float(np.nanmean(p)) if p else np.nan)

    return (
        np.array(empirical_sigma_mean),
        np.array(empirical_sigma_window),
        {k: np.array(v) for k, v in pred_sigma_means.items()}
    )

# ------------------------------------------------------------------ #
#  Plot
# ------------------------------------------------------------------ #
def main():
    print(f"Running Monte Carlo  (N={N_SIGNALS} signals) ...", flush=True)
    
    # Run simulation
    (sigma_means_emp, 
     sigma_windows_emp, 
     sigma_means_pred) = compute(BANDWIDTHS, TIME_POINTS, N_SIGNALS,
                                 T_GENERATE, DT, SEED)
    
    # We know the global sigma is 1.0 (by construction in generate_multiband_noise)
    GLOBAL_SIGMA = 1.0
    analytic_sigma_mean = analytic_mixture(TIME_POINTS, BANDWIDTHS) * GLOBAL_SIGMA

    # 1. Standard Plot: Normalized Sigma_Mean / Global_Sigma
    #    (This is essentially identical to the raw plot for unit-variance signals,
    #     but labeled as a ratio)
    plot_metric(
        FIG_DIR / "multiband_comparison.png",
        TIME_POINTS,
        target_curve=analytic_sigma_mean / GLOBAL_SIGMA,
        empirical_curve=sigma_means_emp / GLOBAL_SIGMA,
        pred_curves={k: v / GLOBAL_SIGMA for k, v in sigma_means_pred.items()},
        ylabel=r"$\sigma_\mathrm{mean}\,/\,\sigma_\mathrm{signal,global}$",
        title="Normalized Standard Deviation of the Mean"
    )

    # 2. Raw Plot: Sigma_Mean (not normalized)
    plot_metric(
        FIG_DIR / "multiband_sigma_mean.png",
        TIME_POINTS,
        target_curve=analytic_sigma_mean,
        empirical_curve=sigma_means_emp,
        pred_curves=sigma_means_pred,
        ylabel=r"$\sigma_\mathrm{mean}$",
        title="Raw Standard Deviation of the Mean"
    )

    # 3. Window Sigma Ratio: Sigma_Window / Global_Sigma
    plot_metric(
        FIG_DIR / "multiband_sigma_window.png",
        TIME_POINTS,
        target_curve=None,
        empirical_curve=sigma_windows_emp / GLOBAL_SIGMA,
        pred_curves=None,
        ylabel=r"$\sigma_\mathrm{window}\,/\,\sigma_\mathrm{signal,global}$",
        title="Ratio of Window Std Dev to Global Std Dev",
    )


def plot_metric(filename, time_points, target_curve, empirical_curve, pred_curves, ylabel, title):
    fig, ax = plt.subplots(figsize=(10, 6))

    if target_curve is not None:
        if isinstance(target_curve, (list, np.ndarray)):
            ax.loglog(time_points, target_curve, "k-", linewidth=2.5,
                  label="Analytic target", zorder=5)

    if empirical_curve is not None:
        ax.loglog(time_points, empirical_curve, "k--", linewidth=1.5, alpha=0.7,
                  label=f"Empirical (N={N_SIGNALS})", zorder=4)

    if pred_curves is not None:
        for lbl, _ in METHODS:
            if lbl in pred_curves:
                color, marker, ls, lw = METHOD_STYLE[lbl]
                ax.plot(time_points, pred_curves[lbl],
                        color=color, marker=marker, linestyle=ls, linewidth=lw,
                        markersize=5, alpha=0.9, label=lbl)

    ax.set_xlabel("Integration time  t  (s)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.set_title(f"{title}\nMulti-band noise  B = {' + '.join(str(b) for b in BANDWIDTHS)} Hz", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  -> {filename}")



if __name__ == "__main__":
    main()
