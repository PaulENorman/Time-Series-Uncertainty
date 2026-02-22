"""
Run the canonical MOCKET figure/table scripts in order.

Usage:
    python workflows/reproduce.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "figure_00_signal_and_running_mean.py",
    "figure_01_effective_sample_size.py",
    "figure_02_ergodic_batches.py",
    "figure_03_obm_overlap_diagram.py",
    "figure_04_batch_means_intuition.py",
    "figure_05_bl_functional_form.py",
    "figure_06_beyond_bm_methods.py",
    "figure_07_underresolved_methods.py",
    "figure_08_midresolved_methods.py",
    "figure_09_overresolved_methods.py",
    "figure_10_performance_vs_B.py",
    "figure_11_joint_decomposition.py",
    "plot_eval.py",
    "plot_joint_fit_B1.py",
    "plot_joint_sigma_extrapolation_B1.py",
    "plot_joint_decay_sweep_B1.py",
    "plot_multiband.py",
    "test_eval.py",
]


def main() -> int:
    for script in SCRIPTS:
        print(f"\n=== Running {script} ===", flush=True)
        proc = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        if proc.returncode != 0:
            print(f"Failed: {script} (exit {proc.returncode})", flush=True)
            return proc.returncode
    print("\nAll scripts completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
