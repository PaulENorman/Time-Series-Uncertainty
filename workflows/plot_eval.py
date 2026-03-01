"""
Evaluation helpers for comparing estimator predictions against
the analytic BL uncertainty curve across resolution regimes.

Because the analytic target is exact, a modest number of realizations
is sufficient for stable comparison curves.

Produces:
    docs/assets/figures/eval_signature_<name>.png
    docs/assets/figures/eval_regime_summary.png

Usage:
    python workflows/plot_eval.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
WORKFLOWS_DIR = Path(__file__).resolve().parent
if str(WORKFLOWS_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOWS_DIR))

try:
    from estimators import var_mean_bl
    from method_registry import METHODS, METHOD_STYLE
    from paths import FIG_DIR
    from noise import generate_bl_noise
except ImportError:
    from mocket.estimators import var_mean_bl
    from mocket.workflows.method_registry import METHODS, METHOD_STYLE
    from mocket.workflows.paths import FIG_DIR
    from mocket.noise import generate_bl_noise

DT = 2e-4
T_GENERATE = 300.0
N_SIGNALS = 40          # increased for lower-noise estimator averages
TIME_POINTS = np.logspace(np.log10(0.2), np.log10(100.0), 20)
FREQ_RES = 1.0 / 100.0  # = 0.01 Hz (based on T_max = 100 s)
EVAL_BASE_SEED = 20260223

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

def _b_key(B: float) -> int:
    """Stable integer key for deterministic per-bandwidth seeding."""
    return int(round(float(B) * 1_000_000_000))


def _generate_signal_bank(B, n_signals, t_generate, dt, seed):
    """Deterministic signal bank for a given (B, seed, n_signals)."""
    b_key = _b_key(B)
    ss = np.random.SeedSequence([int(seed), b_key, 101])
    child_seqs = ss.spawn(int(n_signals))
    signals = []
    for i, cseq in enumerate(child_seqs):
        finite_sig = None
        # Retry with deterministic offsets if generation is non-finite.
        for attempt in range(6):
            s = int(cseq.generate_state(1, dtype=np.uint32)[0] + attempt)
            sig = generate_bl_noise(B, t_generate, dt, seed=s)
            if np.all(np.isfinite(sig)):
                finite_sig = sig
                break
        if finite_sig is None:
            raise RuntimeError(f"Failed to generate finite signal for B={B}, idx={i}")
        signals.append(finite_sig)
    return signals


def _sample_starts(B, time_idx, max_start, n_signals, seed):
    """Deterministic start indices per (B, time_idx, seed)."""
    b_key = _b_key(B)
    ss = np.random.SeedSequence([int(seed), b_key, 202, int(time_idx)])
    rng = np.random.default_rng(ss)
    return rng.integers(0, max_start, size=int(n_signals), endpoint=False)


def compute_curves(B, time_points=TIME_POINTS, n_signals=N_SIGNALS,
                   t_generate=T_GENERATE, dt=DT, seed=EVAL_BASE_SEED,
                   estimators=None):
    """
    For each time point return the analytic ratio and the mean predicted
    ratio for each method over n_signals realisations.
    """
    n_long = int(round(t_generate / dt))
    methods = METHODS if estimators is None else list(estimators.items())
    signals = _generate_signal_bank(B, n_signals=n_signals, t_generate=t_generate, dt=dt, seed=seed)

    analytic_curve = []
    pred_curves = {lbl: [] for lbl, _ in methods}

    for ti, t in enumerate(time_points):
        n_eval = int(round(t / dt))
        max_start = max(1, n_long - n_eval)
        starts = _sample_starts(B, ti, max_start=max_start, n_signals=n_signals, seed=seed)
        analytic_curve.append(float(np.sqrt(var_mean_bl(t, B))))

        for lbl, fn in methods:
            preds = []
            for sig, start in zip(signals, starts):
                start = int(start)
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
    for xv, c in [(0.9, "red"), (3, "orange"), (100, "green")]:
        ax.axvline(xv, color=c, linestyle=":", linewidth=1)
    y_text = max(1e-4, np.nanmax([np.nanmax(v) for v in method_maes.values()]) * 0.9)
    ax.text(0.3, y_text, "under", color="red", fontsize=8, ha="center")
    ax.text(1.6, y_text, "near", color="darkorange", fontsize=8, ha="center")
    ax.text(17, y_text, "resolved", color="green", fontsize=8, ha="center")
    ax.text(2000, y_text, "over", color="blue", fontsize=8, ha="center")

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
