import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from paths import FIG_DIR
except ImportError:
    from mocket.workflows.paths import FIG_DIR


def main():
    t = np.logspace(-1, 2, 30)
    y_true = 0.35 * t ** (-0.5)
    rng = np.random.default_rng(7)
    y_obs = y_true * np.exp(0.09 * rng.standard_normal(t.size))

    k = 10
    x = np.log(t[-k:])
    z = np.log(y_obs[-k:])
    m, c = np.polyfit(x, z, 1)
    m = float(np.clip(m, -0.5, 0.0))
    y_fit = np.exp(c) * t ** m

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.loglog(t, y_obs, "ko", ms=4, alpha=0.75, label="Observed tail points")
    ax.loglog(t, y_true, "k--", lw=1.4, label=r"Reference $t^{-1/2}$")
    ax.loglog(t, y_fit, color="tab:green", lw=2.0, label=f"Bounded tail fit (slope={m:.2f})")
    ax.axvline(t[-k], color="0.35", ls=":", lw=1.0)
    ax.text(t[-k] * 1.05, y_obs.max() * 0.85, "Tail fit region", fontsize=9)
    ax.set_xlabel("Integration time t (s)")
    ax.set_ylabel(r"$\sigma_{\mathrm{mean}}/\sigma_{\mathrm{signal}}$")
    ax.set_title("Tail-fit theory: bounded log-linear extrapolation")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    out = FIG_DIR / "figure_11_tail_theory.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
