"""
Bandwidth-limited noise generation.
"""

import numpy as np
import scipy.fft as sfft


def noise_from_pow_spec(rfft_freqs, pow_spec, seed=None):
    """
    Generate noise whose power spectrum matches *pow_spec*.

    Parameters
    ----------
    rfft_freqs : 1-D ndarray
        Real-FFT frequency grid.
    pow_spec : 1-D ndarray
        Target power spectrum (same length as *rfft_freqs*).
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    ndarray (1-D) — time-domain noise realisation.
    """
    rng = np.random.default_rng(seed)
    rand_phase = np.array(np.sqrt(pow_spec), dtype="complex")
    phi = 2 * np.pi * rng.random(rfft_freqs.size)
    phi = np.cos(phi) + 1j * np.sin(phi)
    rand_phase *= phi
    return sfft.irfft(rand_phase).T


def band_limited_noise(f_c, bw, length, dt, mag_std=1.0, seed=None):
    """
    Generate bandwidth-limited white noise.

    Parameters
    ----------
    f_c : float
        Centre frequency (Hz).
    bw : float
        Bandwidth (Hz).
    length : float
        Signal duration in seconds (rounded down to nearest *dt*).
    dt : float
        Sampling time step (s).
    mag_std : float
        Desired standard deviation of the output signal.
    seed : int, optional
        RNG seed.

    Returns
    -------
    ndarray (1-D)
    """
    min_freq = f_c - 0.5 * bw
    max_freq = f_c + 0.5 * bw
    sig_len = int(length / dt)

    freqs = sfft.rfftfreq(sig_len, dt)
    p_spec = np.zeros_like(freqs)

    mask = (freqs > min_freq) & (freqs < max_freq)
    p_spec[mask] = 1.0

    df = freqs[1] - freqs[0]
    norm = dt * np.sqrt(2 * df * np.sum(p_spec)) / np.sqrt(length)

    return mag_std * noise_from_pow_spec(freqs, p_spec, seed=seed) / norm


def generate_bl_noise(bandwidth, length, dt, mean=0.0, seed=None):
    """
    Convenience wrapper: generate band-limited noise with passband
    [0, *bandwidth*] Hz, unit variance, shifted to the given *mean*.

    Parameters
    ----------
    bandwidth : float
        One-sided bandwidth B (Hz).
    length : float
        Signal duration (s).
    dt : float
        Time step (s).
    mean : float
        Desired signal mean.
    seed : int, optional

    Returns
    -------
    ndarray (1-D)
    """
    sig = band_limited_noise(
        f_c=bandwidth / 2.0, bw=bandwidth,
        length=length, dt=dt, mag_std=1.0, seed=seed,
    )
    return sig + mean


def generate_multiband_noise(bandwidths, length, dt, seed=None):
    """
    Generate a sum of independent BL noise components, one per bandwidth,
    normalised to unit variance overall.  Using multiple bandwidths prevents
    the estimator from being evaluated only on signals that perfectly match
    the single-component BL spectral model.

    Parameters
    ----------
    bandwidths : sequence of float
        One-sided bandwidths B_i (Hz) of each component.
    length : float
        Signal duration (s).
    dt : float
        Time step (s).
    seed : int, optional
        Base RNG seed.  Each component uses seed + i so realisations are
        independent but reproducible.

    Returns
    -------
    ndarray (1-D) — unit-variance, zero-mean composite signal.
    """
    rng = np.random.default_rng(seed)
    component_seeds = rng.integers(0, 2**31, size=len(bandwidths))

    parts = [
        generate_bl_noise(B, length, dt, seed=int(component_seeds[i]))
        for i, B in enumerate(bandwidths)
    ]
    sig = np.sum(parts, axis=0)
    sig -= np.mean(sig)
    std = np.std(sig)
    return sig / std if std > 0 else sig


def generate_piecewise_power_noise(break_freq, slope, length, dt, seed=None):
    """
    Generate zero-mean, unit-std noise with piecewise PSD:
      - constant for 0 < f <= break_freq
      - proportional to (f / break_freq)^slope for f > break_freq

    Parameters
    ----------
    break_freq : float
        Break frequency in Hz.
    slope : float
        Power-law exponent above break frequency (e.g. -5/3).
    length : float
        Signal duration in seconds.
    dt : float
        Time step in seconds.
    seed : int, optional
        RNG seed.

    Returns
    -------
    ndarray (1-D) — zero-mean, unit-variance signal.
    """
    n = int(length / dt)
    freqs = sfft.rfftfreq(n, dt)
    p_spec = np.ones_like(freqs, dtype=float)
    p_spec[0] = 0.0

    b = max(float(break_freq), 1e-12)
    mask_hi = freqs > b
    p_spec[mask_hi] = np.power(freqs[mask_hi] / b, float(slope))
    p_spec = np.clip(p_spec, 0.0, np.inf)

    sig = noise_from_pow_spec(freqs, p_spec, seed=seed)
    sig = np.asarray(sig, dtype=float)
    sig -= np.mean(sig)
    std = float(np.std(sig))
    return sig / std if std > 0 else sig
