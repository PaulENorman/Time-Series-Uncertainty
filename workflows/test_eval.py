"""
Evaluate MOCKET estimators against a purely empirical target.

Evaluation methodology
----------------------
For N independent signal realisations and a given window length t:
  1. Compute the window mean  μ_i  for each trial i.
  2. Compute the window std   σ_i  for each trial i.
  3. Target ratio  =  std({μ_i}) / mean({σ_i})

This is compared to MOCKET predictions from the shared estimator
registry, averaged over the same N trials.

Signals are generated as sums of independent BL noise components
(multi-band) so the estimator is not evaluated on signals that perfectly
match the single-component BL spectral model.

Usage
-----
    python workflows/test_eval.py
"""

import numpy as np

from mocket.workflows.method_registry import METHODS
from mocket.noise import generate_multiband_noise

# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #
DT = 2e-4
N_TRIALS = 200
T_GENERATE = 300.0      # generate long signals; slice windows from within

# Time points at which to evaluate (seconds)
EVAL_TIMES = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]

# Bandwidth combinations to evaluate — mixing components tests generalisation
SIGNATURES = {
    "single  B=0.1":    [0.1],
    "single  B=15":     [15.0],
    "multi   B=0.1+1":  [0.1, 1.0],
    "multi   B=1+15":   [1.0, 15.0],
    "multi   B=0.1+15": [0.1, 15.0],
}


# ------------------------------------------------------------------ #
#  Evaluation
# ------------------------------------------------------------------ #

def evaluate_signature(bandwidths, eval_times=EVAL_TIMES,
                        n_trials=N_TRIALS, t_generate=T_GENERATE,
                        dt=DT, seed=0):
    """
    Evaluate all methods against the empirical target ratio at each
    evaluation time.  Returns a list of dicts, one per time point.
    """
    n_long = int(round(t_generate / dt))
    rng = np.random.default_rng(seed)

    signals = [
        generate_multiband_noise(bandwidths, t_generate, dt,
                                 seed=seed * 100_000 + i)
        for i in range(n_trials)
    ]

    rows = []
    for t in eval_times:
        n_eval = int(round(t / dt))
        max_start = max(1, n_long - n_eval)

        trial_means, trial_stds = [], []
        method_preds = [[] for _ in METHODS]

        for sig in signals:
            start = int(rng.integers(0, max_start))
            w = sig[start: start + n_eval]
            trial_means.append(float(np.mean(w)))
            trial_stds.append(float(np.std(w)))
            vs = np.var(w)
            if vs > 0:
                for mi, (_, fn) in enumerate(METHODS):
                    est = fn(w)
                    if est > 0:
                        method_preds[mi].append(float(np.sqrt(est / vs)))

        target = float(np.std(trial_means) / np.mean(trial_stds))
        row = dict(t=t, target=target)
        for mi, (label, _) in enumerate(METHODS):
            preds = method_preds[mi]
            mean = float(np.nanmean(preds)) if preds else float("nan")
            row[label] = mean
            row[f"{label}_err%"] = (
                (mean - target) / target * 100 if target > 0 and np.isfinite(mean)
                else float("nan")
            )
        rows.append(row)

    return rows


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    method_labels = [label for label, _ in METHODS]
    # widths: t, target, then per-method pred+err pairs
    w_val, w_err = 10, 9

    def _fmt(v, width=w_val):
        return f"{v:.5f}" if np.isfinite(v) else "  nan  "

    def _fmt_err(v, width=w_err):
        return f"{v:+.2f}%" if np.isfinite(v) else "  nan  "

    for sig_name, bandwidths in SIGNATURES.items():
        print(f"\n{'='*80}")
        print(f"  {sig_name}  (B = {bandwidths},  N = {N_TRIALS} trials)")
        print(f"{'='*80}")
        # header
        hdr = f"{'t (s)':>8}  {'target':>{w_val}}"
        for lbl in method_labels:
            hdr += f"  {lbl:>{w_val}}  {'err%':>{w_err}}"
        print(hdr)
        print("-" * len(hdr))

        rows = evaluate_signature(bandwidths)
        for r in rows:
            line = f"{r['t']:8.2f}  {_fmt(r['target'])}"
            for lbl in method_labels:
                line += f"  {_fmt(r[lbl])}  {_fmt_err(r[f'{lbl}_err%'])}"
            print(line)
