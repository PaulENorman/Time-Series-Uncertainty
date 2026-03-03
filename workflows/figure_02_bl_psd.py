import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from paths import FIG_DIR

B = 1.0


def main():
    f = np.logspace(-3, 2, 900)
    psd = np.where(f <= B, 1.0, 1e-14)

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.loglog(f, psd, color="tab:blue", lw=2.1, label="Band-limited white noise")
    ax.axvline(B, color="tab:red", ls="--", lw=1.3, label=f"Bandwidth B={B:g} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Band-limited reference PSD")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_02_bl_psd.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
