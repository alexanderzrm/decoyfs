"""CLI: python run_benchmark.py --tier smoke|laptop|full [--force-gpu] [--out results/xxx]

Requires ``pip install -e ".[bench]"``. The ``full`` tier reproduces the DecoyFS paper (74 datasets, 5x2 CV).
"""
import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))

from decoyfs.bench.data import MAXN_SENS, TIERS  # noqa: E402
from decoyfs.bench.presets import SENS_CFG, SENS_MAXN, TIER_CFG  # noqa: E402
from decoyfs.bench.runner import DEFAULT_CFG, run_all  # noqa: E402
from decoyfs.device import device, env_report  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="smoke", choices=list(TIERS))
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--force-gpu", action="store_true")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--stab-B", type=int)
    ap.add_argument("--no-cvk", action="store_true")
    ap.add_argument("--no-ext", action="store_true", help="disable the extensions of the full tier")
    ap.add_argument("--sensitivity", action="store_true", help="after the run, execute the max_n sensitivity analysis")
    a = ap.parse_args()
    device(force_gpu=a.force_gpu)
    cfg = dict(TIER_CFG[a.tier])
    if a.no_ext:
        for k in ("ablations", "baselines", "pass2", "model_plan"):
            cfg.pop(k, None)
    if a.models:
        cfg["models"] = a.models
    if a.methods:
        cfg["methods"] = a.methods
    if a.stab_B is not None:
        cfg["stab_B"] = a.stab_B
    if a.no_cvk:
        cfg["cvk"] = False
    out = a.out or os.path.join(HERE, "results", a.tier)
    os.makedirs(out, exist_ok=True)
    env = env_report()
    print(json.dumps(env, indent=1))
    with open(os.path.join(out, "env.json"), "w") as f:
        json.dump(env, f, indent=2)
    from decoyfs.bench.progress import Tracker  # noqa: E402
    cfg_main = {**DEFAULT_CFG, **cfg}
    jobs = [(a.datasets or TIERS[a.tier], out, cfg_main, "main")]
    if a.sensitivity:
        jobs += [(MAXN_SENS, os.path.join(HERE, "results", f"{a.tier}_maxn{mx}"), {**DEFAULT_CFG, **SENS_CFG, "max_n": mx}, "sens")
                 for mx in SENS_MAXN]
    trk = Tracker(jobs, ui=False)
    run_all(a.datasets or TIERS[a.tier], out, cfg_main, tracker=trk)
    if a.sensitivity:
        for mx in SENS_MAXN:
            run_all(MAXN_SENS, os.path.join(HERE, "results", f"{a.tier}_maxn{mx}"), {**DEFAULT_CFG, **SENS_CFG, "max_n": mx}, tracker=trk)
