"""Shared method/style registry for article figures."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from estimators import var_mean_bl_fit
except ImportError:
    from mocket.estimators import var_mean_bl_fit

WORKFLOWS_DIR = Path(__file__).resolve().parent
if str(WORKFLOWS_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOWS_DIR))
from methods_bm_obm_acc_tail import (
    var_mean_bm,
    var_mean_obm50,
    var_mean_obm75,
    var_mean_acc_unbiased,
    var_mean_acc_zero_corrected,
    var_mean_tail,
)


def _var_mean_obm(signal, overlap):
    return float(var_mean_bl_fit(signal, overlapping=True, overlap_ratio=float(overlap)))


CORE_METHODS = {
    "BM": var_mean_bm,
    "OBM 50%": var_mean_obm50,
    "ACC-unbiased": var_mean_acc_unbiased,
    "Tail": var_mean_tail,
}


CORE_METHODS_ORDERED = list(CORE_METHODS.items())


HYBRID_METHODS = {
    "BM": var_mean_bm,
    "OBM 75%": var_mean_obm75,
    "ACC-0c": var_mean_acc_zero_corrected,
    "Tail": var_mean_tail,
}


HYBRID_METHOD_STYLE = {
    "BM": ("tab:blue", "o", "-", 1.9),
    "OBM 75%": ("tab:orange", "s", "--", 1.9),
    "ACC-0c": ("tab:purple", "D", ":", 1.9),
    "Tail": ("tab:green", "^", "-.", 1.9),
}


OBM_SWEEP_METHODS = {
    "BM": var_mean_bm,
    "OBM 10%": lambda x: _var_mean_obm(x, 0.10),
    "OBM 25%": lambda x: _var_mean_obm(x, 0.25),
    "OBM 50%": lambda x: _var_mean_obm(x, 0.50),
    "OBM 75%": lambda x: _var_mean_obm(x, 0.75),
    "OBM 99%": lambda x: _var_mean_obm(x, 0.99),
}


CORE_METHOD_STYLE = {
    "BM": ("tab:blue", "o", "-", 1.8),
    "OBM 50%": ("tab:orange", "s", "--", 1.8),
    "ACC-unbiased": ("tab:purple", "D", "-.", 1.8),
    "Tail": ("tab:green", "^", ":", 1.8),
}


OBM_SWEEP_STYLE = {
    "BM": ("tab:blue", "o", "-", 1.8),
    "OBM 10%": ("tab:olive", "v", "--", 1.8),
    "OBM 25%": ("tab:cyan", "<", "--", 1.8),
    "OBM 50%": ("tab:orange", "s", "--", 1.8),
    "OBM 75%": ("tab:pink", ">", "--", 1.8),
    "OBM 99%": ("tab:red", "P", "--", 1.8),
}

# Backward compatibility for older scripts.
METHODS = CORE_METHODS_ORDERED
METHOD_STYLE = CORE_METHOD_STYLE
