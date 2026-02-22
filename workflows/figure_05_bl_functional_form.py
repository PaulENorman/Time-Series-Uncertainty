import numpy as np
import matplotlib.pyplot as plt

from mocket.estimators import var_mean_bl
from mocket.workflows.paths import FIG_DIR


def main():
    t = np.logspace(np.log10(0.02), np.log10(120.0), 400)
    bands = [0.02, 0.1, 1.0, 10.0]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for b in bands:
        y = np.sqrt(var_mean_bl(t, b))
        ax.loglog(t, y, lw=2.0, label=f"B={b:g} Hz")

    t_ref = np.logspace(np.log10(2.0), np.log10(120.0), 120)
    y_ref = 0.7 * t_ref ** (-0.5)
    ax.loglog(t_ref, y_ref, "k--", lw=1.8, label=r"$\propto t^{-1/2}$ guide")

    ax.set_xlabel("Integration time t (s)")
    ax.set_ylabel(r"$\sigma_{\mathrm{mean}}/\sigma_{\mathrm{signal}} = \sqrt{\mathrm{var\_mean\_bl}(t,B)}$")
    ax.set_title("Figure 5 — BL functional form and asymptotic scaling")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figure_05_bl_functional_form.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
