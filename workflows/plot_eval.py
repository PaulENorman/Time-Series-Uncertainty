"""
Log-log plots comparing the shared MOCKET estimator set
(BM, OBM, BM-std, BM-joint, BM-dual)
to the analytic BL formula across the full resolution spectrum.

Because the analytic target is exact, we only need a small number of
signal realisations to get a stable mean prediction — much faster than
building an empirical target from hundreds of trials.

Produces:
    docs/assets/figures/eval_signature_<name>.png  — ratio vs time for each bandwidth
    docs/assets/figures/eval_regime_summary.png    — MAE vs resolution factor, all methods

Usage:
    python workflows/plot_eval.py
"""

import numpy as np
import matplotlib.pyplot as plt

from mocket.estimators import (
    var_mean_bl,
)
from mocket.workflows.method_registry import METHODS, METHOD_STYLE
from mocket.workflows.paths import FIG_DIR
from mocket.noise import generate_bl_noise

DT = 2e-4
T_GENERATE = 300.0
N_SIGNALS = 20          # small — only needed to average out estimator noise
TIME_POINTS = np.logspace(np.log10(0.2), np.log10(100.0), 20)
FREQ_RES = 1.0 / 100.0  # = 0.01 Hz (based on T_max = 100 s)

# (display name, bandwidth, resolution factor B/f_res)
SIGNATURES = [
    ("B=0.005 (×0.5 under)",  0.005,  0.5),
    ("B=0.008 (×0.8 under)",  0.008,  0.8),
    ("B=0.012 (×1.2 near)",   0.012,  1.2),
    ("B=0.030 (×3 near)",     0.030,  3.0),
    ("B=0.10  (×10 res)",     0.10,   10.),
    ("B=1.0   (×100 res)",    1.0,    100.),
    ("B=15    (×1500 over)",  15.0,   1500.),
    ("B=40    (×4000 over)",  40.0,   4000.),
]

def compute_curves(B, time_points=TIME_POINTS, n_signals=N_SIGNALS,
                   t_generate=T_GENERATE, dt=DT, seed=0):
    """
    For each time point return the analytic ratio and the mean predicted
    ratio for each method over n_signals realisations.
    """
    n_long = int(round(t_generate / dt))
    rng = np.random.default_rng(seed)

    signals = [
        generate_bl_noise(B, t_generate, dt, seed=seed * 10000 + i)
        for i in range(n_signals)
    ]

    analytic_curve = []
    pred_curves = {lbl: [] for lbl, _ in METHODS}

    for t in time_points:
        n_eval = int(round(t / dt))
        max_start = max(1, n_long - n_eval)
        analytic_curve.append(float(np.sqrt(var_mean_bl(t, B))))

        for lbl, fn in METHODS:
            preds = []
            for sig in signals:
                start = int(rng.integers(0, max_start))
                w = sig[start: start + n_eval]
                vs = np.var(w)
                if vs > 0:
                    est = fn(w)
                    if est > 0:
                        preds.append(float(np.sqrt(est / vs)))
            pred_curves[lbl].append(float(np.nanmean(preds)) if preds else np.nan)

    return np.array(analytic_curve), {k: np.array(v) for k, v in pred_curves.items()}


def plot_signature(name, B, res_factor):
    print(f"  {name}", flush=True)
    analytic, preds = compute_curves(B)

    resolution_label = (
        "underresolved" if res_factor < 0.9
        else "near-limit" if res_factor < 3
        else "overresolved" if res_factor > 100
        else "resolved"
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(TIME_POINTS, analytic, "k-", linewidth=2.5, label="Analytic target")

    for lbl, _ in METHODS:
        color, marker, ls, lw = METHOD_STYLE[lbl]
        ax.plot(TIME_POINTS, preds[lbl],
                color=color, marker=marker, linestyle=ls, linewidth=lw,
                markersize=4, alpha=0.9, label=lbl)

    ax.set_xlabel("Integration time  t  (s)")
    ax.set_ylabel(r"$\sigma_\mathrm{mean}\,/\,\sigma_\mathrm{signal}$")
    ax.set_title(f"{name}   [{resolution_label},  B/f_res = {res_factor:.1f}]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    safe = name.split("(")[0].strip().replace("=", "").replace(".", "p").replace(" ", "_")
    out = FIG_DIR / f"eval_signature_{safe}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"    -> {out}")


def plot_regime_summary():
    """MAE vs resolution factor for all methods."""
    print("  Computing regime summary ...", flush=True)
    res_factors, method_maes = [], {lbl: [] for lbl, _ in METHODS}

    for name, B, res_factor in SIGNATURES:
        analytic, preds = compute_curves(B)
        valid = analytic > 0
        res_factors.append(res_factor)
        for lbl, _ in METHODS:
            p = preds[lbl][valid]
            a = analytic[valid]
            mae = float(np.nanmean(np.abs((p - a) / a)))
            method_maes[lbl].append(mae)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.array(res_factors)

    for lbl, _ in METHODS:
        color, marker, ls, lw = METHOD_STYLE[lbl]
        ax.semilogx(x, method_maes[lbl], color=color, marker=marker,
                    linestyle=ls, linewidth=lw, markersize=8, alpha=0.9, label=lbl)

    ax.axvspan(0.01, 0.9,   alpha=0.07, color="red")
    ax.axvspan(0.9,  3,     alpha=0.07, color="orange")
    ax.axvspan(3,    100,   alpha=0.07, color="green")
    ax.axvspan(100,  1e5,   alpha=0.07, color="blue")
    for xv, c, txt in [(0.9, "red", "under"), (3, "orange", "near"),
                        (100, "green", "resolved"), (5000, "blue", "over")]:
        ax.axvline(xv if xv < 100 else 100, color=c, linestyle=":", linewidth=1)
    ax.text(0.3,   ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1, "under",   color="red",    fontsize=8, ha="center")
    ax.text(1.6,   0, "near",    color="darkorange", fontsize=8, ha="center")
    ax.text(17,    0, "resolved",color="green",      fontsize=8, ha="center")
    ax.text(2000,  0, "over",    color="blue",        fontsize=8, ha="center")

    ax.set_xlabel("Resolution factor  B / f_res  (= B · T_max)")
    ax.set_ylabel("Mean absolute relative error vs analytic")
    ax.set_title(f"Method accuracy across resolution regimes  (N={N_SIGNALS} signals)")
    ax.set_xlim(min(x) * 0.4, max(x) * 2.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=10)

    out = FIG_DIR / "eval_regime_summary.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"    -> {out}")


if __name__ == "__main__":
    print("Generating per-signature plots ...")
    for name, B, res_factor in SIGNATURES:
        plot_signature(name, B, res_factor)

    print("Generating regime summary ...")
    plot_regime_summary()

    print("Done.")

