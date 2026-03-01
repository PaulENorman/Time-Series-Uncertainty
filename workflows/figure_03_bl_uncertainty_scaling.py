import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from estimators import var_mean_bl
from paths import FIG_DIR


def main():
    t = np.logspace(np.log10(0.02), np.log10(120.0), 500)
    bands = [0.1, 1.0, 10.0]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for b in bands:
        y = np.sqrt(var_mean_bl(t, b))
        ax.loglog(t, y, lw=2.0, label=f"B={b:g} Hz")

    tref = np.logspace(np.log10(2.0), np.log10(120.0), 150)
    yref = 0.7 * tref ** (-0.5)
    ax.loglog(tref, yref, "k--", lw=1.6, label=r"$\propto t^{-1/2}$ guide")

    ax.set_xlabel("Integration time t (s)")
    ax.set_ylabel(r"$\sigma_{\mathrm{mean}}/\sigma_{\mathrm{signal}}$")
    ax.set_title("Analytic BL uncertainty scaling")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_03_bl_uncertainty_scaling.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
