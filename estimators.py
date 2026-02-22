"""
Variance-of-the-mean estimators based on batch means and
bandwidth-limited (BL) curve fitting.
"""

import numpy as np
from numpy import pi, sin, log
from scipy.special import sici
from scipy.optimize import curve_fit


# ------------------------------------------------------------------ #
#  Batch-means primitives                                             #
# ------------------------------------------------------------------ #

def bm_var_mean(signal, batch_len):
    """
    Divide *signal* into non-overlapping batches of length *batch_len*
    and return the variance of the batch means.
    """
    rem = signal.size % batch_len
    nb = int(signal.size / batch_len)
    if rem == 0:
        batches = np.split(signal, nb)
    else:
        batches = np.split(signal[:-rem], nb)
    return np.var(np.average(batches, 1))


def obm_mean_var(signal, batch_len, batch_overlap):
    """
    Divide *signal* into overlapping batches and return the variance of
    the batch means.
    """
    cumsum = np.cumsum(np.insert(signal, 0, 0))
    mavg = (cumsum[batch_len:] - cumsum[:-batch_len]) / float(batch_len)

    if batch_len <= batch_overlap:
        raise ValueError("batch_overlap must be less than batch_len")
    step = batch_len - batch_overlap
    return np.var(mavg[::step])


def _overlap_samples(batch_len, overlap_ratio):
    """Convert overlap ratio to a valid overlap sample count."""
    ratio = float(overlap_ratio)
    if not (0.0 <= ratio < 1.0):
        raise ValueError("overlap_ratio must be in [0, 1)")
    overlap = int(ratio * batch_len)
    return min(overlap, max(0, int(batch_len) - 1))


# ------------------------------------------------------------------ #
#  Analytic bandwidth-limited variance formulas                       #
# ------------------------------------------------------------------ #

def var_bl(t, B):
    """Variance correction factor: 1 − var_mean_bl(t, B)."""
    return 1.0 - var_mean_bl(t, B)


def var_mean_bl(t, B):
    """
    Normalised variance of the sample mean for a BL signal of duration
    *t* and one-sided bandwidth *B*.
    """
    return (-(sin(B * pi * t)) ** 2 + B * pi * t * sici(2 * B * pi * t)[0]) / (
        (B * pi * t) ** 2
    )


def log_var_mean_bl(t, B):
    """Log of var_mean_bl — used as the fitting objective."""
    return log(var_mean_bl(t, B))


# ------------------------------------------------------------------ #
#  MKT estimators (BL curve-fit)                                      #
# ------------------------------------------------------------------ #

def var_mean_bl_fit(signal, vc=False, min_b=20, fit_b=10,
                    overlapping=False, overlap_ratio=0.5):
    """
    Estimate Var(X̄) by fitting the analytic BL variance curve to
    batch-mean variances at several batch sizes.

    Parameters
    ----------
    signal : 1-D array
    vc : bool
        Apply finite-sample variance correction.
    min_b : int
        Minimum number of batches at the largest batch size.
    fit_b : int
        Number of log-spaced batch sizes for the fit.
    overlapping : bool
        If True, use overlapping batch means (reduces sampling noise in
        the fitted bandwidth).  If False (default), use non-overlapping
        batch means.
    overlap_ratio : float
        Fraction of batch length used as overlap (0 … <1).  Only used
        when *overlapping* is True.

    Returns
    -------
    float — estimated Var(X̄)
    """
    var_sig = np.var(signal)
    if len(signal) < 20:
        return var_sig

    bl_end = int(signal.size / min_b)

    if overlapping:
        ov0 = _overlap_samples(bl_end, overlap_ratio)
        var_est = obm_mean_var(signal, bl_end, ov0) / var_sig
    else:
        var_est = bm_var_mean(signal, bl_end) / var_sig

    B_est = 1.0 / (2 * var_est * bl_end)
    bl_start = 1.0 / (2 * B_est)

    batches = np.logspace(np.log10(bl_start), np.log10(bl_end), fit_b).astype(int)
    batches = np.unique(batches[batches != 0])
    varb = np.zeros(len(batches))

    for i, b in enumerate(batches):
        if overlapping:
            ov = _overlap_samples(b, overlap_ratio)
            varb[i] = obm_mean_var(signal, b, ov)
        else:
            varb[i] = bm_var_mean(signal, b)
    varb /= var_sig

    try:
        B = curve_fit(
            log_var_mean_bl,
            xdata=batches,
            ydata=log(varb),
            p0=B_est,
            bounds=(0, np.inf),
        )[0][0]
    except Exception:
        B = B_est

    if vc:
        var_corr = var_bl(len(signal), B)
    else:
        var_corr = 1

    return var_mean_bl(len(signal), B) * var_sig / var_corr


# ------------------------------------------------------------------ #
#  Batch-stds estimator                                               #
# ------------------------------------------------------------------ #

def var_mean_bl_std_fit(signal, min_b=20, fit_b=10):
    """
    Estimate Var(X̄) by fitting the BL curve to the *mean batch standard
    deviation* at several batch sizes rather than the variance of batch means.

    For a BL signal of bandwidth B, the expected batch std at batch length b is:

        E[std(batch)] ≈ σ · sqrt(var_bl(b, B))
                      = σ · sqrt(1 − var_mean_bl(b, B))

    This is an *increasing* function of b (rising from 0 towards σ).  At
    early/unresolved times (b << 1/(2B)) the batch-mean variance is nearly
    flat at 1, giving the standard estimator little curvature to fit.  The
    batch-std curve starts near zero and has significant slope in the same
    regime, providing much stronger leverage on B.

    Parameters
    ----------
    signal : 1-D array
    min_b : int
        Minimum number of batches at the largest batch size.
    fit_b : int
        Number of log-spaced batch sizes for the fit.

    Returns
    -------
    float — estimated Var(X̄)
    """
    signal = np.asarray(signal, dtype=float)
    n = signal.size
    var_sig = np.var(signal)
    if n < 20 or var_sig <= 0:
        return var_sig

    bl_end = int(n / min_b)
    var_est = bm_var_mean(signal, bl_end) / var_sig
    B_est = max(1e-9, 1.0 / (2.0 * var_est * bl_end))
    bl_start = max(2, int(1.0 / (2.0 * B_est)))

    batches = np.logspace(np.log10(bl_start), np.log10(bl_end), fit_b).astype(int)
    batches = np.unique(batches[batches >= 2])

    valid_b, valid_v = [], []
    for b in batches:
        b = int(b)
        nb = n // b
        if nb < 2:
            continue
        segs = signal[: nb * b].reshape(nb, b)
        # Use mean within-batch variance (ddof=1), not mean std.
        # E[s²_ddof1] = (b/(b-1)) · σ² · var_bl(b, B)  — exact in expectation,
        # no Jensen correction needed.
        mean_var = float(np.mean(np.var(segs, axis=1, ddof=1)))
        val = mean_var / var_sig  # estimates (b/(b-1)) · var_bl(b, B)
        if val > 0:
            valid_b.append(float(b))
            valid_v.append(val)

    if len(valid_b) < 3:
        # Fall back to standard fit if insufficient valid points
        return var_mean_bl_fit(signal, min_b=min_b, fit_b=fit_b)

    valid_b = np.array(valid_b)
    valid_v = np.array(valid_v)

    def log_within_var_bl(t, B):
        """log((t/(t-1)) · var_bl(t, B)) — exact expectation of s²/σ²."""
        return log(t / (t - 1.0)) + log(np.clip(var_bl(t, B), 1e-15, None))

    try:
        B = curve_fit(
            log_within_var_bl,
            xdata=valid_b,
            ydata=np.log(valid_v),
            p0=B_est,
            bounds=(1e-12, np.inf),
        )[0][0]
    except Exception:
        B = B_est

    return var_mean_bl(n, B) * var_sig


# ------------------------------------------------------------------ #
#  Asymptotic variance correction / joint fit                         #
# ------------------------------------------------------------------ #

def var_mean_bl_joint_fit(signal, min_b=20, fit_b=10, overlapping=False,
                          overlap_ratio=0.5, sigma2_clip_lo=0.5,
                          sigma2_clip_hi=2.0,
                          sigma_decay_rate=6.0,
                          sigma_clip_lo=0.8,
                          sigma_clip_hi=1.25):
    """
    Jointly fit global variance and bandwidth from *absolute* batch-mean
    variances, then predict Var(X̄) using the fitted asymptotic variance.

    Model
    -----
    For batch length b,
        Var(batch_mean_b) = sigma_inf^2 * var_mean_bl(b, B)

    where sigma_inf^2 is the long-time (asymptotic) process variance.
    Unlike :func:`var_mean_bl_fit`, this does not normalise by the local
    window variance, so it can recover missing low-frequency variance when
    the window is too short.
    """
    signal = np.asarray(signal, dtype=float)
    n = signal.size
    var_sig = np.var(signal)
    if n < 20 or var_sig <= 0:
        return var_sig

    bl_end = int(n / min_b)
    if bl_end < 2:
        return var_sig

    # Seed B with the standard fit's local ratio heuristic.
    if overlapping:
        ov0 = _overlap_samples(bl_end, overlap_ratio)
        var_est = obm_mean_var(signal, bl_end, ov0) / max(var_sig, 1e-15)
    else:
        var_est = bm_var_mean(signal, bl_end) / max(var_sig, 1e-15)
    B_est = max(1e-12, 1.0 / (2.0 * max(var_est, 1e-12) * bl_end))

    bl_start = max(2, int(1.0 / (2.0 * B_est)))
    batches = np.logspace(np.log10(bl_start), np.log10(bl_end), fit_b).astype(int)
    batches = np.unique(batches[(batches >= 2) & (batches <= bl_end)])
    if batches.size < 3:
        return var_mean_bl_fit(signal, min_b=min_b, fit_b=fit_b, overlapping=overlapping, overlap_ratio=overlap_ratio)

    varb = np.zeros(len(batches), dtype=float)
    for i, b in enumerate(batches):
        if overlapping:
            ov = _overlap_samples(int(b), overlap_ratio)
            varb[i] = obm_mean_var(signal, int(b), ov)
        else:
            varb[i] = bm_var_mean(signal, int(b))

    valid = np.isfinite(varb) & (varb > 0)
    if valid.sum() < 3:
        return var_mean_bl_fit(signal, min_b=min_b, fit_b=fit_b, overlapping=overlapping, overlap_ratio=overlap_ratio)

    x = batches[valid].astype(float)
    y = np.log(varb[valid])

    def log_abs_var_model(t, log_sigma2, B):
        return log_sigma2 + log_var_mean_bl(t, B)

    log_sigma2_0 = float(np.log(max(var_sig, 1e-15)))
    try:
        popt = curve_fit(
            log_abs_var_model,
            xdata=x,
            ydata=y,
            p0=(log_sigma2_0, B_est),
            bounds=((-40.0, 1e-12), (40.0, np.inf)),
        )[0]
        sigma2_inf = float(np.exp(popt[0]))
        B = float(popt[1])
    except Exception:
        sigma2_inf = var_sig
        B = B_est

    # Mild stabilization: keep asymptotic variance within a broad,
    # physically plausible range around the local window variance.
    # This damps occasional noisy over/under-shoot from the 2-parameter fit
    # while preserving the intended low-frequency correction.
    if sigma2_clip_lo is not None and sigma2_clip_hi is not None:
        lo = max(1e-15, float(sigma2_clip_lo) * var_sig)
        hi = max(lo, float(sigma2_clip_hi) * var_sig)
        sigma2_inf = float(np.clip(sigma2_inf, lo, hi))

    # Regime-aware shrinkage of the sigma correction:
    # correction factor -> 1 exponentially as B*n grows (well-resolved).
    # unresolved (B*n << 1): retain most of fitted correction.
    bn = max(0.0, float(B) * float(n))
    keep = float(np.exp(-float(sigma_decay_rate) * bn))
    sigma2_inf = float(var_sig + keep * (sigma2_inf - var_sig))

    # Optional explicit limiter in sigma-space (std-dev), relative to the
    # observed window sigma. This is often more intuitive than sigma^2 bounds.
    if sigma_clip_lo is not None or sigma_clip_hi is not None:
        sigma_win = float(np.sqrt(max(var_sig, 1e-15)))
        sigma_inf = float(np.sqrt(max(sigma2_inf, 1e-15)))

        lo = 0.0 if sigma_clip_lo is None else max(0.0, float(sigma_clip_lo)) * sigma_win
        hi = np.inf if sigma_clip_hi is None else max(lo, float(sigma_clip_hi) * sigma_win)
        sigma_inf = float(np.clip(sigma_inf, lo, hi))
        sigma2_inf = float(sigma_inf * sigma_inf)

    return sigma2_inf * var_mean_bl(n, B)


def var_mean_bl_dual_fit(signal, min_b=20, fit_b=10, overlapping=False,
                         overlap_ratio=0.5, within_weight=1.0):
    """
    Dual-fit estimator using both BM and within-batch variance curves.

    Fits shared (sigma_inf^2, B) from two observables across batch size b:

      1) Var(batch_mean_b)             = sigma_inf^2 * var_mean_bl(b, B)
      2) E[s2_within_b, ddof=1]        = sigma_inf^2 * (b/(b-1)) * var_bl(b, B)

    This combines the leverage of BM (good in scaling regime) and the
    within-batch variance trend (informative in flatter regimes).
    """
    signal = np.asarray(signal, dtype=float)
    n = signal.size
    var_sig = np.var(signal)
    if n < 20 or var_sig <= 0:
        return var_sig

    bl_end = int(n / min_b)
    if bl_end < 2:
        return var_sig

    if overlapping:
        ov0 = _overlap_samples(bl_end, overlap_ratio)
        var_est = obm_mean_var(signal, bl_end, ov0) / max(var_sig, 1e-15)
    else:
        var_est = bm_var_mean(signal, bl_end) / max(var_sig, 1e-15)
    B_est = max(1e-12, 1.0 / (2.0 * max(var_est, 1e-12) * bl_end))

    bl_start = max(2, int(1.0 / (2.0 * B_est)))
    batches = np.logspace(np.log10(bl_start), np.log10(bl_end), fit_b).astype(int)
    batches = np.unique(batches[(batches >= 2) & (batches <= bl_end)])
    if batches.size < 3:
        return var_mean_bl_joint_fit(
            signal,
            min_b=min_b,
            fit_b=fit_b,
            overlapping=overlapping,
            overlap_ratio=overlap_ratio,
        )

    bm_obs = []
    wv_obs = []
    valid_b = []
    for b in batches:
        b = int(b)
        if overlapping:
            ov = _overlap_samples(b, overlap_ratio)
            vmean = obm_mean_var(signal, b, ov)
        else:
            vmean = bm_var_mean(signal, b)

        nb = n // b
        if nb < 2:
            continue
        segs = signal[: nb * b].reshape(nb, b)
        vwithin = float(np.mean(np.var(segs, axis=1, ddof=1)))

        if np.isfinite(vmean) and np.isfinite(vwithin) and vmean > 0 and vwithin > 0:
            bm_obs.append(float(vmean))
            wv_obs.append(float(vwithin))
            valid_b.append(float(b))

    if len(valid_b) < 3:
        return var_mean_bl_joint_fit(
            signal,
            min_b=min_b,
            fit_b=fit_b,
            overlapping=overlapping,
            overlap_ratio=overlap_ratio,
        )

    b = np.array(valid_b, dtype=float)
    y_bm = np.log(np.array(bm_obs, dtype=float))
    y_wv = np.log(np.array(wv_obs, dtype=float))

    x_t = np.concatenate([b, b])
    x_kind = np.concatenate([np.zeros_like(b), np.ones_like(b)])
    y = np.concatenate([y_bm, y_wv])

    def log_dual_model(xdata, log_sigma2, B):
        t = xdata[0]
        kind = xdata[1]
        vmean = np.clip(var_mean_bl(t, B), 1e-15, None)
        vbl = np.clip(var_bl(t, B), 1e-15, None)
        pred_bm = log_sigma2 + np.log(vmean)
        pred_wv = log_sigma2 + np.log(t / (t - 1.0)) + np.log(vbl)
        return np.where(kind < 0.5, pred_bm, pred_wv)

    miss_frac0 = float(np.clip(var_mean_bl(n, B_est), 1e-6, 0.999))
    sigma2_0 = max(var_sig / (1.0 - miss_frac0), var_sig, 1e-15)
    log_sigma2_0 = float(np.log(sigma2_0))

    sigmas = np.ones_like(y)
    if within_weight > 0:
        sigmas[len(b):] = 1.0 / float(within_weight)

    try:
        popt = curve_fit(
            log_dual_model,
            xdata=np.vstack([x_t, x_kind]),
            ydata=y,
            p0=(log_sigma2_0, B_est),
            sigma=sigmas,
            bounds=((-40.0, 1e-12), (40.0, np.inf)),
            absolute_sigma=False,
        )[0]
        sigma2_inf = float(np.exp(popt[0]))
        B = float(popt[1])
    except Exception:
        return var_mean_bl_joint_fit(
            signal,
            min_b=min_b,
            fit_b=fit_b,
            overlapping=overlapping,
            overlap_ratio=overlap_ratio,
        )

    return sigma2_inf * var_mean_bl(n, B)
