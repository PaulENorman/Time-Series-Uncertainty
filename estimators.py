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
    x = np.asarray(signal, dtype=float)
    b = int(batch_len)
    if b <= 0:
        raise ValueError("batch_len must be positive")
    nb = x.size // b
    if nb < 1:
        return float(np.var(x))
    trim = x[: nb * b]
    means = trim.reshape(nb, b).mean(axis=1)
    return float(np.var(means))


def obm_mean_var(signal, batch_len, batch_overlap):
    """
    Divide *signal* into overlapping batches and return the variance of
    the batch means.
    """
    b = int(batch_len)
    ov = int(batch_overlap)
    if b <= ov:
        raise ValueError("batch_overlap must be less than batch_len")
    cumsum = _prefix_sum(signal)
    return _obm_mean_var_from_prefix(cumsum, b, ov)


def _prefix_sum(signal):
    """Length-(n+1) prefix sum with cumsum[0] = 0."""
    x = np.asarray(signal, dtype=float)
    cumsum = np.empty(x.size + 1, dtype=float)
    cumsum[0] = 0.0
    np.cumsum(x, out=cumsum[1:])
    return cumsum


def _obm_mean_var_from_prefix(cumsum, batch_len, batch_overlap):
    """OBM variance using a precomputed prefix sum array."""
    if batch_len <= batch_overlap:
        raise ValueError("batch_overlap must be less than batch_len")
    mavg = (cumsum[batch_len:] - cumsum[:-batch_len]) / float(batch_len)
    step = batch_len - batch_overlap
    return float(np.var(mavg[::step]))


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
    signal = np.asarray(signal, dtype=float)
    var_sig = float(np.var(signal))
    if len(signal) < 20 or var_sig <= 0:
        return var_sig

    bl_end = max(2, int(signal.size / min_b))

    cumsum = _prefix_sum(signal) if overlapping else None

    if overlapping:
        ov0 = _overlap_samples(bl_end, overlap_ratio)
        var_est = _obm_mean_var_from_prefix(cumsum, bl_end, ov0) / var_sig
    else:
        var_est = bm_var_mean(signal, bl_end) / var_sig

    B_est = max(1e-12, 1.0 / (2.0 * max(var_est, 1e-12) * bl_end))
    bl_start = max(2.0, 1.0 / (2.0 * B_est))

    batches = np.logspace(np.log10(bl_start), np.log10(bl_end), fit_b).astype(int)
    batches = np.unique(batches[(batches >= 2) & (batches <= bl_end)])
    if batches.size < 3:
        return float(var_mean_bl(len(signal), B_est) * var_sig)
    varb = np.zeros(len(batches))

    for i, b in enumerate(batches):
        if overlapping:
            ov = _overlap_samples(b, overlap_ratio)
            varb[i] = _obm_mean_var_from_prefix(cumsum, b, ov)
        else:
            varb[i] = bm_var_mean(signal, b)
    varb /= max(var_sig, 1e-15)
    valid = np.isfinite(varb) & (varb > 0)
    if valid.sum() < 3:
        return float(var_mean_bl(len(signal), B_est) * var_sig)

    try:
        B = curve_fit(
            log_var_mean_bl,
            xdata=batches[valid],
            ydata=log(varb[valid]),
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
#  Autocovariance (ACC) estimator                                     #
# ------------------------------------------------------------------ #

def _autocov_fft(signal, unbiased=False, max_lag=50000):
    """FFT-based autocovariance sequence up to *max_lag*."""
    x = np.asarray(signal, dtype=float)
    n = x.size
    x = x - np.mean(x)
    nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    fx = np.fft.rfft(x, n=nfft)
    c_full = np.fft.irfft(fx * np.conj(fx), n=nfft)
    c = c_full[:n]
    if unbiased:
        denom = np.arange(n, 0, -1, dtype=float)
        c = c / np.maximum(denom, 1.0)
    else:
        c = c / float(n)
    max_lag = min(n - 1, int(max_lag))
    return c[: max_lag + 1]


def _truncate_n_zero_crossings(c, n_crossings=1):
    """Keep autocovariance terms up to the n-th non-positive crossing."""
    if c.size <= 1:
        return c
    neg_idx = np.where(c[1:] <= 0.0)[0]
    if neg_idx.size < int(n_crossings):
        return c
    cut = int(neg_idx[int(n_crossings) - 1] + 1)
    return c[: cut + 1]


def var_mean_acc_unbiased(signal):
    """
    Estimate Var(X̄) from unbiased autocovariance integration with
    first-zero-crossing truncation.
    """
    x = np.asarray(signal, dtype=float)
    n = x.size
    if n < 5:
        return float(np.var(x))
    c = _autocov_fft(x, unbiased=True)
    c = _truncate_n_zero_crossings(c, n_crossings=1)
    # ACC-zero: integrate raw autocovariance up to first zero crossing.
    var_mean = (c[0] + 2.0 * np.sum(c[1:])) / float(n)
    return float(max(var_mean, 1e-15))


# For BL noise, truncating ACC at first zero crossing overpredicts asymptotic
# variance by: R_var = 2 * Si(pi) / pi  (continuous-time derivation).
ACC_ZERO_BL_VAR_OVERPRED = float(2.0 * sici(pi)[0] / pi)   # ~1.17898
ACC_ZERO_BL_VAR_CORRECTION = float(1.0 / ACC_ZERO_BL_VAR_OVERPRED)  # ~0.84819


def var_mean_acc_unbiased_bl_corrected(signal):
    """
    ACC-zero with BL-analytic bias correction.

    For Cxx(tau)=sigma^2*sinc(2*pi*B*tau), first-zero truncation at
    tau0=1/(2B) gives:
        Var_est / Var_true -> 2*Si(pi)/pi  as T->infinity
    so we scale by its inverse.
    """
    return float(max(var_mean_acc_unbiased(signal) * ACC_ZERO_BL_VAR_CORRECTION, 1e-15))


def var_mean_acc_tail_damped(signal, lag_fraction=0.25):
    """
    ACC-tail damp variant:
        C_b(k) = (1 - k/n) C(k)
    with a practical finite-record lag cap.

    In discrete finite windows, integrating the biased estimate across all
    lags can over-cancel and collapse toward zero. We therefore cap the lag
    range to a fraction of record length, consistent with using only the
    reliably resolved low-frequency part of the autocovariance.
    """
    x = np.asarray(signal, dtype=float)
    n = x.size
    if n < 5:
        return float(np.var(x))
    max_lag = int(np.clip(float(lag_fraction) * float(n), 1, n - 1))
    c = _autocov_fft(x, unbiased=True, max_lag=max_lag)
    k = np.arange(c.size, dtype=float)
    c_b = (1.0 - (k / float(n))) * c
    var_mean = (c_b[0] + 2.0 * np.sum(c_b[1:])) / float(n)
    return float(max(var_mean, 1e-15))


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
    cumsum = _prefix_sum(signal) if overlapping else None

    if overlapping:
        ov0 = _overlap_samples(bl_end, overlap_ratio)
        var_est = _obm_mean_var_from_prefix(cumsum, bl_end, ov0) / max(var_sig, 1e-15)
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
            varb[i] = _obm_mean_var_from_prefix(cumsum, int(b), ov)
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
