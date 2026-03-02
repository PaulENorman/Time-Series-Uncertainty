import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from noise import generate_piecewise_power_noise
    from stabilization import (
        add_exponential_transient,
        estimate_stabilization_time_mockett,
    )
except ImportError:
    from mocket.noise import generate_piecewise_power_noise
    from mocket.stabilization import (
        add_exponential_transient,
        estimate_stabilization_time_mockett,
    )


DT = 2e-4
T_SIGNAL = 6.0
T_GENERATE = 3.0 * T_SIGNAL
BREAK_FREQ = 10.0
TAIL_SLOPE = -5.0 / 3.0
TRUE_T_STAR = 1.0
CUTOFF_FRAC = 0.10
AMP_LEVELS = [1.0, 2.0, 3.0]
N_PER_AMP = 120
BASE_SEED = 20260321


def _build_signal(seed, amp_std):
    x_long = generate_piecewise_power_noise(
        break_freq=BREAK_FREQ,
        slope=TAIL_SLOPE,
        length=T_GENERATE,
        dt=DT,
        seed=seed,
    )
    n = int(round(T_SIGNAL / DT))
    rng = np.random.default_rng(seed + 100)
    start = int(rng.integers(0, max(1, len(x_long) - n)))
    x = np.asarray(x_long[start : start + n], dtype=float)
    x_w_tr, _ = add_exponential_transient(
        x,
        dt=DT,
        amplitude_std=float(amp_std),
        cutoff_time=TRUE_T_STAR,
        cutoff_fraction=CUTOFF_FRAC,
    )
    return x, x_w_tr


def _detect_cusum_mean_cp(signal):
    x = np.asarray(signal, dtype=float)
    n = x.size
    max_i = int(0.5 * n)
    tail_n = max(512, int(0.25 * n))
    mu_ref = float(np.mean(x[-tail_n:]))
    y = x[:max_i] - mu_ref
    c = np.cumsum(y)
    i = int(np.argmax(np.abs(c)))
    return float(i * DT)


def _detect_rolling_zmeanvar(signal, z_thresh=2.0, std_rel=0.12, consec=3):
    x = np.asarray(signal, dtype=float)
    n = x.size
    w = max(512, int(round(0.4 / DT)))  # 0.4 s window
    step = max(32, int(round(0.05 / DT)))  # 0.05 s stride
    if n < 3 * w:
        return np.nan
    ref = x[-w:]
    mu_r = float(np.mean(ref))
    var_r = float(np.var(ref, ddof=1))
    std_r = max(float(np.std(ref, ddof=1)), 1e-12)
    run = 0
    cand_t = np.nan
    for i in range(0, int(0.5 * n), step):
        if i + w >= n - w:
            break
        seg = x[i : i + w]
        mu = float(np.mean(seg))
        var = float(np.var(seg, ddof=1))
        std = float(np.std(seg, ddof=1))
        se = np.sqrt(max(var / w + var_r / w, 1e-18))
        z = abs(mu - mu_r) / se
        rel_std = abs(std - std_r) / std_r
        if (z <= z_thresh) and (rel_std <= std_rel):
            if run == 0:
                cand_t = i * DT
            run += 1
            if run >= int(consec):
                return float(cand_t)
        else:
            run = 0
            cand_t = np.nan
    return np.nan


def _detect_reverse_mean_band(signal, z_band=2.0, consec=3):
    x = np.asarray(signal, dtype=float)
    n = x.size
    if n < 1024:
        return np.nan
    tail_n = max(1024, int(0.25 * n))
    ref = x[-tail_n:]
    mu_ref = float(np.mean(ref))
    var_ref = float(np.var(ref, ddof=1))
    run = 0
    cand = np.nan
    step = max(32, int(round(0.05 / DT)))
    min_len = max(1024, int(round(0.5 / DT)))
    for i in range(0, int(0.5 * n), step):
        seg = x[i:]
        m = seg.size
        if m < min_len:
            break
        mu = float(np.mean(seg))
        var = float(np.var(seg, ddof=1))
        se = np.sqrt(max(var / m + var_ref / tail_n, 1e-18))
        z = abs(mu - mu_ref) / se
        if z <= z_band:
            if run == 0:
                cand = i * DT
            run += 1
            if run >= int(consec):
                return float(cand)
        else:
            run = 0
            cand = np.nan
    return np.nan


def _trim_bias(x_base, x, t_star):
    if not np.isfinite(t_star):
        return np.nan
    i = int(np.clip(round(float(t_star) / DT), 0, len(x) - 1))
    return float(np.mean(x[i:]) - np.mean(x_base))


def _summarize(arr):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(a)), float(np.mean(np.abs(a))), float(np.std(a))


def main():
    methods = {
        "Mockett_ACC0c": lambda sig: float(
            estimate_stabilization_time_mockett(
                sig, dt=DT, max_remove_fraction=0.5, min_segment_samples=256
            ).t_star
        ),
        "CUSUM_mean_cp": _detect_cusum_mean_cp,
        "Rolling_ZMeanVar": _detect_rolling_zmeanvar,
        "ReverseMeanBand": _detect_reverse_mean_band,
    }

    terr = {k: [] for k in methods}
    mbias = {k: [] for k in methods}
    hit02 = {k: [] for k in methods}

    for amp in AMP_LEVELS:
        for i in range(N_PER_AMP):
            x_base, x = _build_signal(BASE_SEED + 10000 * int(amp) + i, amp_std=amp)
            for name, fn in methods.items():
                t_hat = float(fn(x))
                e = t_hat - TRUE_T_STAR if np.isfinite(t_hat) else np.nan
                terr[name].append(e)
                mbias[name].append(_trim_bias(x_base, x, t_hat))
                hit02[name].append(float(abs(e) <= 0.2) if np.isfinite(e) else np.nan)

    print(f"Stationarity benchmark (N={N_PER_AMP} per amp, amps={AMP_LEVELS})")
    print("Method | mean(t_hat-1.0) [s] | MAE_t [s] | std_t [s] | mean bias | MAE bias | hit(|dt|<=0.2s)")
    for name in methods:
        m_e, mae_e, std_e = _summarize(terr[name])
        m_b, mae_b, _ = _summarize(mbias[name])
        m_h, _, _ = _summarize(hit02[name])
        print(
            f"{name:16s} | {m_e:+.4f} | {mae_e:.4f} | {std_e:.4f} | "
            f"{m_b:+.5f} | {mae_b:.5f} | {m_h:.3f}"
        )


if __name__ == "__main__":
    main()
