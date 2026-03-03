"""Generate the ordered article figure set used in docs/index.md."""

from pathlib import Path
import subprocess
import sys


ORDERED_SCRIPTS = [
    "figure_01_signal_running_mean.py",       # -> figure_01_signal_running_mean.png
    "figure_02_bl_psd.py",                    # -> figure_02_bl_psd.png
    "figure_03_bl_uncertainty_scaling.py",    # -> figure_03_bl_uncertainty_scaling.png
    "figure_03b_raw_bl_signals.py",           # -> figure_03b_raw_bl_signals.png
    "figure_04_bm_intuition.py",              # -> figure_04_bm_intuition.png
    "figure_05_obm_overlap_diagram.py",       # -> figure_05_obm_overlap_diagram.png
    "figure_06_obm_vs_bm_regimes.py",        # -> figure_06_obm_vs_bm_regimes.png
    "figure_07_obm_vs_bm_mae.py",            # -> figure_07_obm_vs_bm_mae.png
    "figure_08_acc_theory.py",                # -> figure_08_acc_theory.png
    "figure_09_acc_variants_single_B.py",    # -> figure_09_acc_variants_single_B.png
    "figure_10_acc_variants_mae_vs_B.py",    # -> figure_10_acc_variants_mae_vs_B.png
    "figure_11_tail_theory.py",               # -> figure_11_tail_theory.png
    "figure_12_tail_vs_bm_regimes.py",       # -> figure_12_tail_vs_bm_regimes.png
    "figure_13_tail_vs_bm_mae.py",           # -> figure_13_tail_vs_bm_mae.png
    "figure_14_methods_regimes.py",           # -> figure_14_methods_regimes.png
    "figure_15_methods_mae_vs_B.py",          # -> figure_15_methods_mae_vs_B.png
    "figure_16_realworld_noise.py",           # -> figure_16_realworld_noise.png
    "figure_17_18_realworld_combined.py",     # -> figure_17_realworld_methods.png + figure_18_realworld_mae_vs_time.png
    "figure_19_stationarity_mockett.py",      # -> figure_19_stationarity_mockett.png
    "figure_20_stationarity_distribution.py", # -> figure_20_stationarity_distribution.png
]


def main():
    workflows_dir = Path(__file__).resolve().parent
    for script in ORDERED_SCRIPTS:
        path = workflows_dir / script
        print(f"[fig] {script}", flush=True)
        subprocess.run([sys.executable, str(path)], check=True)


if __name__ == "__main__":
    main()
