"""Generate the ordered article figure set used in docs/index.md."""

from pathlib import Path
import subprocess
import sys


ORDERED_SCRIPTS = [
    "figure_01_signal_running_mean.py",       # -> figure_01_signal_running_mean.png
    "figure_02_bl_psd.py",                    # -> figure_02_bl_psd.png
    "figure_03_bl_uncertainty_scaling.py",    # -> figure_03_bl_uncertainty_scaling.png
    "figure_04_bm_intuition.py",              # -> figure_04_bm_intuition.png
    "figure_05_obm_overlap_diagram.py",       # -> figure_05_obm_overlap_diagram.png
    "figure_05b_obm_vs_bm_regimes.py",        # -> figure_06_obm_vs_bm_regimes.png
    "figure_05c_obm_vs_bm_mae.py",            # -> figure_07_obm_vs_bm_mae.png
    "figure_06_acc_theory.py",                # -> figure_08_acc_theory.png
    "figure_08b_acc_variants_single_B.py",    # -> figure_08b_acc_variants_single_B.png
    "figure_08c_acc_variants_mae_vs_B.py",    # -> figure_08c_acc_variants_mae_vs_B.png
    "figure_07_tail_theory.py",               # -> figure_09_tail_theory.png
    "figure_09b_tail_vs_bm_regimes.py",       # -> figure_09b_tail_vs_bm_regimes.png
    "figure_09c_tail_vs_bm_mae.py",           # -> figure_09c_tail_vs_bm_mae.png
    "figure_09_methods_regimes.py",           # -> figure_11_methods_regimes.png
    "figure_10_methods_mae_vs_B.py",          # -> figure_12_methods_mae_vs_B.png
    "figure_13_realworld_noise.py",           # -> figure_13_realworld_noise.png
    "figure_14_15_realworld_combined.py",     # -> figure_14_realworld_methods.png + figure_15_realworld_mae_vs_time.png
    "figure_16_stationarity_mockett.py",      # -> figure_16_stationarity_mockett.png
    "figure_17_stationarity_distribution.py", # -> figure_17_stationarity_distribution.png
]


def main():
    workflows_dir = Path(__file__).resolve().parent
    for script in ORDERED_SCRIPTS:
        path = workflows_dir / script
        print(f"[fig] {script}", flush=True)
        subprocess.run([sys.executable, str(path)], check=True)


if __name__ == "__main__":
    main()
