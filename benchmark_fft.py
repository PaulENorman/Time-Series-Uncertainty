
import numpy as np
import time
try:
    import scipy.fft as sfft
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def acc_numpy(x, nfft):
    fx = np.fft.rfft(x, n=nfft)
    c_full = np.fft.irfft(fx * np.conj(fx), n=nfft)
    return c_full

def acc_scipy(x, nfft, workers=1):
    fx = sfft.rfft(x, n=nfft, workers=workers)
    c_full = sfft.irfft(fx * np.conj(fx), n=nfft, workers=workers)
    return c_full

def benchmark():
    N = 1_000_000
    x = np.random.randn(N)
    nfft = 1 << int(np.ceil(np.log2(2 * N - 1)))
    
    # Warm up
    _ = np.fft.rfft(x[:1000], n=2048)
    if HAS_SCIPY:
        _ = sfft.rfft(x[:1000], n=2048)

    print(f"Benchmarking FFT-based ACC for N={N} (NFFT={nfft})")
    
    # NumPy
    start = time.time()
    for _ in range(5):
        _ = acc_numpy(x, nfft)
    print(f"numpy.fft: {(time.time() - start)/5:.4f}s")

    if HAS_SCIPY:
        # SciPy (1 worker)
        start = time.time()
        for _ in range(5):
            _ = acc_scipy(x, nfft, workers=1)
        print(f"scipy.fft (1 worker): {(time.time() - start)/5:.4f}s")
        
        # SciPy (Multiple workers)
        start = time.time()
        for _ in range(5):
            _ = acc_scipy(x, nfft, workers=-1)
        print(f"scipy.fft (multi-worker): {(time.time() - start)/5:.4f}s")
    else:
        print("SciPy not found in environment.")

    # Check for MKL/OpenBLAS
    print("\nEnvironment Info:")
    try:
        np.show_config()
    except:
        pass

if __name__ == "__main__":
    benchmark()
