import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from mocket.workflows.paths import FIG_DIR


def main():
    rng = np.random.default_rng(13)
    n = 2000
    t = np.linspace(0.0, 100.0, n)

    transient = 0.9 * np.exp(-t / 18.0)
    slow_mode = 0.25 * np.sin(2 * np.pi * t / 28.0)
    noise = 0.10 * rng.normal(size=n)
    x = transient + slow_mode + noise

    n_batches = 10
    batch_len = n // n_batches
    x = x[: batch_len * n_batches]
    t = t[: batch_len * n_batches]

    mean_global = float(np.mean(x))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(t, x, color="0.2", linewidth=1.2, label="Signal")

    colors = plt.get_cmap("tab20")(np.linspace(0, 1, n_batches))
    for i in range(n_batches):
        start = i * batch_len
        end = (i + 1) * batch_len
        ts = t[start]
        te = t[end - 1]
        batch_mean = float(np.mean(x[start:end]))
        y_min = np.min(x) - 0.08
        y_max = np.max(x) + 0.08
        rect = Rectangle((ts, y_min), te - ts, y_max - y_min, fill=False, linewidth=1.5, edgecolor=colors[i])
        ax.add_patch(rect)
        ax.text((ts + te) / 2, y_max + 0.01, f"B{i+1}", ha="center", va="bottom", fontsize=8, color=colors[i])
        ax.hlines(batch_mean, ts, te, colors=colors[i], linestyles="-", linewidth=2.4)

    ax.axhline(mean_global, color="tab:blue", linestyle="--", linewidth=2, label=f"Global mean = {mean_global:+.3f}")

    ax.set_xlabel("Time")
    ax.set_ylabel("Signal value")
    ax.set_title("Figure 2 — Ergodic batching idea: one transient record split into 10 batches")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_02_ergodic_batches.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
