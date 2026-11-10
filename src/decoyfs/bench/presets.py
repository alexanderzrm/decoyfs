"""Configuración por nivel (compartida por el notebook y run_benchmark.py)."""
from __future__ import annotations

POL_TUNED = ["ALL", "CTRL_VOTE_MAJ", "FIXK_25", "CVK_1SE", "BORDA_MATCH", "BORUTA", "NABFS"]
POL_EXT = ["ALL", "CTRL_VOTE_MAJ", "CTRL_BLOCK_VOTE_MAJ", "KO_BORDA_q10", "CVK_1SE", "FIXK_25", "BORUTA", "NABFS"]

# Extensiones que abordan las limitaciones 1-5 (ablaciones, baselines, controles con dependencia, modelos
# afinados y de fundación). La limitación 6 (max_n) se evalúa con una corrida aparte sobre MAXN_SENS.
EXT = dict(
    ablations=True, baselines=["BORUTA", "NABFS"], boruta_iter=25, boruta_budget=240.0, nabfs_boot=20, pass2=True,
    model_plan=[dict(models=["logreg_t", "knn_t", "xgb_t", "mlp_t"], policies=POL_TUNED, reps=[0], max_p=1000),
                dict(models=["realmlp"], policies=POL_EXT, reps=[0], max_p=500)],
)

TIER_CFG = {
    "smoke": dict(n_splits=3, n_repeats=1, max_n=5000),
    "laptop": dict(n_splits=5, n_repeats=2, max_n=15000),
    "full": dict(n_splits=5, n_repeats=2, max_n=15000, stab_B=5, **EXT),
}
SENS_MAXN = (3000, 10000)
SENS_CFG = dict(n_splits=5, n_repeats=2)
