import matplotlib.pyplot as plt
import numpy as np

from paths import FIG_DIR


def _draw_bracket(ax, x0, x1, y, h, color, lw=1.8):
    ax.plot([x0, x1], [y, y], color=color, lw=lw)
    ax.plot([x0, x0], [y - h / 2, y + h / 2], color=color, lw=lw)
    ax.plot([x1, x1], [y - h / 2, y + h / 2], color=color, lw=lw)


def main():
    fig, ax = plt.subplots(figsize=(10.4, 4.8), constrained_layout=True)
    total = 20
    width = 4
    starts = list(range(0, total - width + 1, 2))  # 50% overlap

    x = np.linspace(0, total, 900)
    y = 0.25 * np.sin(2 * np.pi * x / 8.0) + 0.08 * np.sin(2 * np.pi * x / 2.2) + 3.0
    ax.plot(x, y, color="0.2", lw=1.5)
    ax.text(-0.1, 3.34, "Signal", fontsize=10, fontweight="bold", color="0.25")

    cmap = plt.get_cmap("tab20")
    for i, s in enumerate(starts, start=1):
        c = cmap((i - 1) / max(1, len(starts) - 1))
        yb = 1.0 + 0.22 * ((i - 1) % 3)
        _draw_bracket(ax, s, s + width, yb, h=0.12, color=c, lw=1.9)
        ax.text(s + width / 2, yb + 0.08, f"O{i}", ha="center", va="bottom", fontsize=8, color=c)

    ax.text(-0.1, 1.56, "Overlapping batches (50% overlap)", fontsize=10, fontweight="bold", color="tab:orange")
    ax.set_xlim(-0.5, total + 0.5)
    ax.set_ylim(0.8, 3.7)
    ax.set_xlabel("Sample index")
    ax.set_yticks([])
    ax.set_title("OBM overlap concept")
    ax.grid(True, axis="x", alpha=0.2)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_05_obm_overlap_diagram.png"
    plt.savefig(out, dpi=170)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
