import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from noise import noise_from_pow_spec
    from paths import FIG_DIR
except ImportError:
    from mocket.noise import noise_from_pow_spec
    from mocket.workflows.paths import FIG_DIR


DT = 2e-4
FS = 1.0 / DT
BREAK_FREQ = 1.0
HI_FREQ_SLOPE = -5.0 / 3.0
T_SIGNAL = 300.0
T_GENERATE = 3.0 * T_SIGNAL
T_SHOW = 20.0
TIME_POINTS = np.logspace(np.log10(0.2), np.log10(100.0), 20)
N_MC_SIGNALS = 80
N_WINDOWS_PER_SIGNAL = 4
BASE_SEED = 260218


def _build_signal(seed):
    n = int(T_SIGNAL / DT)
    n_long = int(T_GENERATE / DT)
    freqs = np.fft.rfftfreq(n_long, DT)
    p_spec = np.ones_like(freqs, dtype=float)
    p_spec[0] = 0.0
    hi = freqs > BREAK_FREQ
    p_spec[hi] = np.power(freqs[hi] / BREAK_FREQ, HI_FREQ_SLOPE)
    x_long = np.asarray(noise_from_pow_spec(freqs, p_spec, seed=seed), dtype=float)
    rng = np.random.default_rng(seed + 5001)
    start = int(rng.integers(0, max(1, n_long - n)))
    x = x_long[start : start + n]
    x -= np.mean(x)
    sx = float(np.std(x))
    return x / sx if sx > 0 else x


def _periodogram(x):
    n = len(x)
    x0 = x - np.mean(x)
    fft = np.fft.rfft(x0)
    f = np.fft.rfftfreq(n, d=DT)
    psd = (np.abs(fft) ** 2) / (FS * n)
    if len(psd) > 2:
        psd[1:-1] *= 2.0
    return f, psd


def _smooth(y, k=300):
    if k <= 1 or len(y) < k:
        return y
    kern = np.ones(k, dtype=float) / float(k)
    return np.convolve(y, kern, mode="same")


def _mc_sigma_ratio(signals, seed):
    n_long = len(signals[0])
    rng = np.random.default_rng(seed)
    y = []
    y_std = []
    for t in TIME_POINTS:
        n_eval = int(round(float(t) / DT))
        max_start = max(1, n_long - n_eval)
        means, sigmas = [], []
        for sig in signals:
            for _ in range(N_WINDOWS_PER_SIGNAL):
                s = int(rng.integers(0, max_start))
                w = sig[s : s + n_eval]
                means.append(float(np.mean(w)))
                sigmas.append(float(np.std(w)))
        means = np.asarray(means, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        sigma_mean = float(np.std(means, ddof=1))
        sigma_signal = float(np.mean(sigmas))
        y.append(sigma_mean / max(sigma_signal, 1e-15))
        y_std.append(float(np.std(np.abs(means), ddof=1) / max(sigma_signal, 1e-15)))
    return np.asarray(y), np.asarray(y_std)


def main():
    x_ref = _build_signal(seed=BASE_SEED)
    mc_signals = [_build_signal(seed=BASE_SEED + 10000 + i) for i in range(N_MC_SIGNALS)]
    ratio_mc, ratio_mc_spread = _mc_sigma_ratio(mc_signals, seed=BASE_SEED + 900000)

    f, psd = _periodogram(x_ref)
    psd_sm = _smooth(psd, k=300)

    n_show = int(round(T_SHOW / DT))
    t_show = np.arange(n_show) * DT

    fig, axs = plt.subplots(3, 1, figsize=(10.2, 11.0))

    ax = axs[0]
    valid = (f > 0) & np.isfinite(psd_sm) & (psd_sm > 0)
    ax.loglog(f[valid], psd_sm[valid], color="tab:blue", lw=1.8)
    ax.axvline(BREAK_FREQ, color="0.45", ls="--", lw=1.0)
    f_ref = np.array([BREAK_FREQ, min(FS / 2.0, 200.0)], dtype=float)
    y_ref = psd_sm[np.argmin(np.abs(f - BREAK_FREQ))] * np.power(f_ref / BREAK_FREQ, HI_FREQ_SLOPE)
    ax.loglog(f_ref, y_ref, "k--", lw=1.4, label=r"$f^{-5/3}$ guide")
    ax.set_xlim(0.02, FS / 2.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Piecewise spectrum: flat to 1 Hz, then $f^{-5/3}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)

    ax = axs[1]
    ax.loglog(TIME_POINTS, ratio_mc, color="tab:red", marker="o", lw=1.8, ms=4.0, label="Monte Carlo mean")
    t_end = float(TIME_POINTS[-1])
    y_end = float(ratio_mc[-1])
    y_t12 = y_end * (TIME_POINTS / t_end) ** (-0.5)
    ax.loglog(TIME_POINTS, y_t12, color="0.25", ls="--", lw=1.6, label=r"$t^{-1/2}$ guide")
    ax.fill_between(
        TIME_POINTS,
        np.clip(ratio_mc - ratio_mc_spread, 1e-12, np.inf),
        np.clip(ratio_mc + ratio_mc_spread, 1e-12, np.inf),
        color="tab:red",
        alpha=0.14,
        linewidth=0,
        label="MC spread",
    )
    ax.set_xlabel("Integration time t (s)")
    ax.set_ylabel(r"$\sigma_{\mathrm{mean}} / \sigma_{\mathrm{signal}}$")
    ax.set_title("Monte Carlo uncertainty trend")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)

    ax = axs[2]
    ax.plot(t_show, x_ref[:n_show], color="0.2", lw=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal")
    ax.set_title("Representative mixed-band time series")
    ax.grid(True, alpha=0.25)

    fig.suptitle("Real-world-style noise: B=1 then $f^{-5/3}$", y=0.995)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_13_realworld_noise.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
