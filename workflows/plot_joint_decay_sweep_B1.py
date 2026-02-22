import numpy as np
import matplotlib.pyplot as plt

from mocket.noise import generate_bl_noise
from mocket.estimators import var_mean_bl, var_mean_bl_joint_fit, var_mean_bl_fit
from mocket.workflows.paths import FIG_DIR

DT = 2e-4
B = 1.0
T_GENERATE = 300.0
TIME_POINTS = np.logspace(np.log10(0.2), np.log10(100.0), 24)
N_SIGNALS = 80
SEED = 123

# Sweep decay rates (higher -> faster collapse to BM-like behavior in resolved regime)
DECAY_VALUES = [0.5, 1.0, 2.0, 4.0, 8.0]

# Two no-decay clip baselines
NO_DECAY_CLIPS = [
    ("no-decay clip_hi=1.2", 0.5, 1.2),
    ("no-decay clip_hi=2.0", 0.5, 2.0),
]


def compute_curve(signals, decay_rate=None, clip_lo=0.5, clip_hi=2.0):
    n_long = int(round(T_GENERATE / DT))
    rng = np.random.default_rng(SEED + 1)
    out = []

    for t in TIME_POINTS:
        n_eval = int(round(t / DT))
        max_start = max(1, n_long - n_eval)
        preds = []

        for sig in signals:
            start = int(rng.integers(0, max_start))
            w = sig[start:start + n_eval]
            vs = np.var(w)
            if vs <= 0:
                continue

            kwargs = dict(
                sigma2_clip_lo=clip_lo,
                sigma2_clip_hi=clip_hi,
            )
            if decay_rate is not None:
                kwargs["sigma_decay_rate"] = decay_rate

            est = var_mean_bl_joint_fit(w, **kwargs)
            if est > 0:
                preds.append(float(np.sqrt(est / vs)))

        out.append(float(np.nanmean(preds)) if preds else np.nan)

    return np.array(out)


def compute_bm_curve(signals):
    n_long = int(round(T_GENERATE / DT))
    rng = np.random.default_rng(SEED + 2)
    out = []

    for t in TIME_POINTS:
        n_eval = int(round(t / DT))
        max_start = max(1, n_long - n_eval)
        preds = []

        for sig in signals:
            start = int(rng.integers(0, max_start))
            w = sig[start:start + n_eval]
            vs = np.var(w)
            if vs <= 0:
                continue
            est = var_mean_bl_fit(w, overlapping=False)
            if est > 0:
                preds.append(float(np.sqrt(est / vs)))

        out.append(float(np.nanmean(preds)) if preds else np.nan)

    return np.array(out)


def main():
    print(f"Generating B={B} signals ...")
    signals = [
        generate_bl_noise(B, T_GENERATE, DT, seed=SEED * 10000 + i)
        for i in range(N_SIGNALS)
    ]

    analytic = np.sqrt(var_mean_bl(TIME_POINTS, B))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(TIME_POINTS, analytic, "k-", lw=2.6, label="Analytic target")

    # Decay sweep with fixed clip
    cmap = plt.get_cmap("viridis")
    for i, d in enumerate(DECAY_VALUES):
        curve = compute_curve(signals, decay_rate=d, clip_lo=0.5, clip_hi=2.0)
        ax.loglog(
            TIME_POINTS,
            curve,
            lw=1.8,
            color=cmap(i / max(1, len(DECAY_VALUES) - 1)),
            label=f"BM-joint decay={d:g} (clip_hi=2.0)",
        )

    # No-decay baselines with different clip highs
    for lbl, clo, chi in NO_DECAY_CLIPS:
        curve = compute_curve(signals, decay_rate=0.0, clip_lo=clo, clip_hi=chi)
        ax.loglog(TIME_POINTS, curve, "--", lw=2.0, label=lbl)

    # BM-joint with no sigma correction at all: force sigma_inf^2 == sigma_window^2
    # by clipping exactly at 1x local variance.
    no_sigma_corr = compute_curve(signals, decay_rate=0.0, clip_lo=1.0, clip_hi=1.0)
    ax.loglog(
        TIME_POINTS,
        no_sigma_corr,
        color="tab:red",
        linestyle="-",
        linewidth=2.2,
        label="BM-joint no sigma correction",
    )

    # Standard BM baseline
    bm_curve = compute_bm_curve(signals)
    ax.loglog(
        TIME_POINTS,
        bm_curve,
        color="tab:orange",
        linestyle=":",
        linewidth=2.2,
        label="BM",
    )

    ax.set_xlabel("Integration time t (s)")
    ax.set_ylabel(r"$\sigma_\mathrm{mean}/\sigma_\mathrm{signal}$")
    ax.set_title(f"B={B}: BM-joint decay sweep and no-decay clip baselines")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "joint_decay_sweep_B1.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
