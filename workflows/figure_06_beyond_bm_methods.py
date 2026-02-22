import numpy as np
import matplotlib.pyplot as plt

from mocket.noise import generate_bl_noise
from mocket.workflows.paths import FIG_DIR


DT = 2e-4
B_TRUE = 15.0
T_WINDOW = 2.4
SEED = 63


def _window_means(x: np.ndarray, b: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, len(x) - b + 1, step, dtype=int)
    means = np.array([np.mean(x[s : s + b]) for s in starts], dtype=float)
    centers = starts + (b // 2)
    return centers, means


def main() -> None:
    n = int(round(T_WINDOW / DT))
    signal = generate_bl_noise(B_TRUE, T_WINDOW, DT, seed=SEED)[:n]
    t = np.arange(len(signal), dtype=float) * DT
    mu = float(np.mean(signal))

    b = max(24, n // 11)
    c_bm, m_bm = _window_means(signal, b=b, step=b)
    c_obm, m_obm = _window_means(signal, b=b, step=max(1, b // 2))

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    ax.plot(t, signal, color="0.7", lw=1.0, alpha=0.9, label="Signal")
    ax.axhline(mu, color="k", ls="--", lw=1.2, alpha=0.8, label="Global mean")

    ax.plot(c_bm * DT, m_bm, "o", ms=6, color="tab:blue", label="BM window means (non-overlap)")
    ax.plot(c_obm * DT, m_obm, "o", ms=4.3, color="tab:orange", alpha=0.85, label="OBM window means (50% overlap)")

    # Draw representative window bars near the center to visualize overlap geometry.
    center = len(signal) // 2
    s0 = max(0, center - 2 * b)
    for k in range(4):
        s = s0 + k * b
        if s + b > len(signal):
            break
        y = np.min(signal) - 0.13 * np.std(signal)
        ax.hlines(y, s * DT, (s + b) * DT, color="tab:blue", lw=2.2, alpha=0.65)
    for k in range(7):
        s = s0 + k * (b // 2)
        if s + b > len(signal):
            break
        y = np.min(signal) - 0.22 * np.std(signal)
        ax.hlines(y, s * DT, (s + b) * DT, color="tab:orange", lw=1.8, alpha=0.6)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal value")
    ax.set_title("Figure 6 — Beyond BM: overlapping windows (OBM) increase averaging density")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8, loc="upper right")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_06_beyond_bm_methods.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
