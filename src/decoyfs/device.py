"""GPU detection for torch and xgboost with explicit diagnostics. Everything falls back to CPU automatically."""
import platform
import subprocess

_STATE = {}


def device(force_gpu: bool = False) -> str:
    """Return 'cuda' if torch can run a real operation on the GPU, otherwise 'cpu'."""
    if "dev" not in _STATE:
        dev, msg = "cpu", ""
        try:
            import torch
            if torch.cuda.is_available():
                a = torch.randn(256, 256, device="cuda")
                float((a @ a).sum())
                dev = "cuda"
            else:
                msg = "torch.cuda.is_available() == False"
        except Exception as e:  # noqa: BLE001
            msg = repr(e)[:300]
        _STATE.update(dev=dev, msg=msg)
    if force_gpu and _STATE["dev"] != "cuda":
        raise RuntimeError(
            "GPU requested (--force-gpu) but torch cannot use it: "
            f"{_STATE['msg']}\nFix: update the NVIDIA driver or install a CUDA build of torch "
            "(see the 'GPU' section of the README)."
        )
    return _STATE["dev"]


def xgb_device() -> str:
    """Return 'cuda' if xgboost can train on the GPU, otherwise 'cpu'."""
    if "xgb" not in _STATE:
        res = "cpu"
        try:
            import numpy as np
            import xgboost as xgb
            if device() == "cuda":
                m = xgb.XGBClassifier(n_estimators=2, tree_method="hist", device="cuda", verbosity=0)
                rng = np.random.default_rng(0)
                m.fit(rng.random((60, 4)), rng.integers(0, 2, 60))
                m.predict(rng.random((5, 4)))
                res = "cuda"
        except Exception as e:  # noqa: BLE001
            _STATE["xgb_msg"] = repr(e)[:200]
        _STATE["xgb"] = res
    return _STATE["xgb"]


def env_report() -> dict:
    import sklearn
    import xgboost
    import torch
    info = dict(python=platform.python_version(), torch=torch.__version__, torch_cuda=torch.version.cuda,
                sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                device=device(), xgb_device=xgb_device(), device_msg=_STATE.get("msg", ""),
                xgb_msg=_STATE.get("xgb_msg", ""))
    if info["device"] == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        info["nvidia_smi"] = out
    except Exception:  # noqa: BLE001
        pass
    return info
