
import numpy as np
import time
from estimators import var_mean_acc_unbiased, var_mean_acc_tail_damped

def profile_acc():
    n = 1_000_000
    signal = np.random.randn(n)
    
    # Warm up
    _ = var_mean_acc_unbiased(signal[:1000])
    
    start = time.time()
    for _ in range(10):
        _ = var_mean_acc_unbiased(signal)
    end = time.time()
    print(f"var_mean_acc_unbiased (N={n}, FFT-based): {(end-start)/10:.4f}s per call")

    start = time.time()
    for _ in range(10):
        _ = var_mean_acc_tail_damped(signal)
    end = time.time()
    print(f"var_mean_acc_tail_damped (N={n}, truncated FFT): {(end-start)/10:.4f}s per call")

if __name__ == "__main__":
    profile_acc()
