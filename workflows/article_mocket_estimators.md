# MOCKET BL Variance Estimators: Finalized Implementation

## Scope
This write-up documents the finalized estimator set and a reproducible workflow.

Final method registry:
- BM
- OBM
- BM-std
- BM-joint

All evaluation scripts import methods from `method_registry.py`, so plots and tables stay consistent.

## 1) Environment
From the repository parent (`/Users/paulnorman/Desktop/arma`):

```bash
cd /Users/paulnorman/Desktop/arma/mocket
source venv/bin/activate
export PYTHONPATH=/Users/paulnorman/Desktop/arma
```

## 2) Generate canonical outputs

```bash
python workflows/reproduce.py
```

This runs:
1. `workflows/plot_eval.py` (single-band signatures + regime summary)
2. `workflows/plot_multiband.py` (multi-band analytic + empirical comparison)
3. `workflows/test_eval.py` (empirical Monte Carlo tables)

## 3) Figure/table outputs
Expected files in `plots/`:
- `eval_signature_*.png`
- `eval_regime_summary.png`
- `multiband_comparison.png`
- `multiband_sigma_mean.png`
- `multiband_sigma_window.png`

Terminal table output:
- printed by `workflows/test_eval.py`

## 4) One-off B=15 ranking table

```bash
python -c "import numpy as np; from mocket.workflows.plot_eval import compute_curves, METHODS; a,p=compute_curves(15.0); rows=[]
for lbl,_ in METHODS:
 v=p[lbl]; m=np.isfinite(v)&np.isfinite(a)&(a>0); rel=np.abs((v[m]-a[m])/a[m]); bias=(v[m]-a[m])/a[m]; rows.append((lbl,float(np.mean(rel)),float(np.sqrt(np.mean((v[m]-a[m])**2))),float(np.mean(bias)),float(np.quantile(rel,0.9))))
rows=sorted(rows,key=lambda r:r[1]); print('Method        MAE(rel)   RMSE(abs)   MeanBias(rel)   P90(rel)');
[print(f'{r[0]:10s}  {r[1]:8.4f}   {r[2]:9.5f}   {r[3]:+12.4f}   {r[4]:8.4f}') for r in rows]; print(f'BEST_BY_MAE: {rows[0][0]} ({rows[0][1]:.4f})')"
```

## 5) Notes
- `var_mean_bl_std_fit` keeps the within-batch variance correction path.
- `var_mean_bl_joint_fit` is retained for asymptotic-variance correction behavior.
- Deprecated experimental variants (`regime`, `refσ`, `hybrid`, `ivc`, legacy aliases) are removed from the package public API.
- Additional exploratory scripts (`plot_joint_*.py`) are unchanged and optional.
