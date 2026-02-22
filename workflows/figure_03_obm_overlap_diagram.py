import matplotlib.pyplot as plt
import numpy as np

from mocket.workflows.paths import FIG_DIR


def _draw_bracket(ax, x0, x1, y, height, color, lw=1.8):
    ax.plot([x0, x1], [y, y], color=color, linewidth=lw)
    ax.plot([x0, x0], [y - height / 2, y + height / 2], color=color, linewidth=lw)
    ax.plot([x1, x1], [y - height / 2, y + height / 2], color=color, linewidth=lw)


def _draw_windows(ax, starts, width, y0, color, label_prefix):
    levels = [0.00, 0.22, 0.44]
    for idx, st in enumerate(starts, start=1):
        y = y0 + levels[(idx - 1) % 3]
        _draw_bracket(ax, st, st + width, y, height=0.12, color=color, lw=1.9)
        ax.text(st + width / 2, y + 0.08, f"{label_prefix}{idx}", ha="center", va="bottom", fontsize=8, color=color)


def main():
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)

    total = 20
    width = 4

    overlap_starts = list(range(0, total - width + 1, 2))

    x = np.linspace(0, total, 800)
    y = 3.15 + 0.22 * np.sin(2 * np.pi * x / 8.0) + 0.08 * np.sin(2 * np.pi * x / 2.2)
    ax.plot(x, y, color="0.25", linewidth=1.5)
    ax.text(-0.2, 3.38, "Signal", fontsize=10, color="0.25", fontweight="bold")

    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(overlap_starts)))
    levels = [0.75, 0.97, 1.19]
    for idx, st in enumerate(overlap_starts, start=1):
        color = colors[idx - 1]
        y = levels[(idx - 1) % 3]
        _draw_bracket(ax, st, st + width, y, height=0.12, color=color, lw=2.0)
        ax.text(st + width / 2, y + 0.08, f"O{idx}", ha="center", va="bottom", fontsize=8, color=color)

        end_x = st + width
        ax.vlines(end_x, ymin=0.28, ymax=1.35, colors=color, linewidth=1.3, alpha=0.9)
        ax.vlines(end_x, ymin=2.82, ymax=3.55, colors=color, linewidth=1.3, alpha=0.9)

    ax.text(-0.2, 1.45, "Overlapping batches (50% overlap)", fontsize=10, color="tab:orange", fontweight="bold")

    ax.set_xlim(-0.5, total + 0.5)
    ax.set_ylim(0.2, 3.7)
    ax.set_xlabel("Sample index")
    ax.set_yticks([])
    ax.set_title("Figure 3 — Overlapping batch windows (bracket view)", pad=16)
    ax.grid(True, axis="x", alpha=0.2)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_03_obm_overlap_diagram.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
