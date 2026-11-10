import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# DecoyFS — Selección de atributos referenciada a controles (Random + Shadow)
Benchmark reproducible sobre datasets públicos (OpenML) y sintéticos con verdad conocida (reproduce los resultados del artículo).
Diseño completo en `docs/DISENO_PROTOCOLO.es.md`. **Portable: usa cualquier kernel Python cuyo torch vea la GPU** (ver `README.md`). Requiere `pip install -e ".[bench]"` desde la raíz del repositorio. Nivel por defecto `full`; para otro nivel define la variable de entorno `DECOYFS_TIER` antes de abrir Jupyter o edita `TIER` abajo.

Flujo: 0) GPU → 1) configuración → 2) ejecución (reanudable) → 3) análisis → 4) exportar.""")

md("## 0a. Verificación automática del entorno (librerías + CUDA/GPU)")
code('''import os, sys, re, json, shutil, subprocess, importlib.metadata as imd
ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
AUTO_INSTALL = True      # instala con pip las librerías de requirements.txt que falten o no cumplan la versión
AUTO_FIX_TORCH = True    # instala/reinstala torch con el build CUDA compatible con el driver si no ve la GPU
SKIP = {"jupyter"}       # el propio notebook ya prueba que Jupyter existe
IMPORT_NAME = {"scikit-learn": "sklearn", "ipykernel": "ipykernel"}
TRUSTED = ["--trusted-host", "pypi.org", "--trusted-host", "files.pythonhosted.org",
           "--trusted-host", "download.pytorch.org", "--trusted-host", "download-r2.pytorch.org"]

def pip(args):
    cmd = [sys.executable, "-m", "pip", "install", *args]
    print("$", " ".join(cmd))
    if subprocess.run(cmd).returncode == 0:
        return True
    print("pip falló; reintentando con --trusted-host (redes con interceptación SSL)")
    return subprocess.run(cmd + TRUSTED).returncode == 0

# ---- librerías de requirements.txt ----
from packaging.requirements import Requirement
def check_requirements():
    bad = []
    for line in open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8"):
        line = line.split("#")[0].strip()
        if not line:
            continue
        r = Requirement(line)
        if r.name.lower() in SKIP:
            continue
        try:
            v = imd.version(r.name)
        except imd.PackageNotFoundError:
            v = None
        ok = v is not None and r.specifier.contains(v, prereleases=True)
        print(f"{'OK ' if ok else 'FALTA'}  {r.name:<14} instalada={v}  requerida={r.specifier or 'cualquiera'}")
        if not ok:
            bad.append(r)
    return bad

bad = check_requirements()
if bad:
    if not AUTO_INSTALL:
        raise RuntimeError(f"Faltan librerías: {[str(r) for r in bad]}. Pon AUTO_INSTALL=True o instala requirements.txt.")
    loaded = [r.name for r in bad if IMPORT_NAME.get(r.name, r.name) in sys.modules]
    if not pip([str(r) for r in bad]):
        raise RuntimeError("No se pudieron instalar las librerías faltantes.")
    if loaded:
        raise RuntimeError(f"Se actualizaron {loaded} que ya estaban cargadas: reinicia el kernel y ejecuta de nuevo desde el inicio.")
    bad = check_requirements()
    assert not bad, f"Siguen sin cumplirse: {[str(r) for r in bad]}"
print("\\nLibrerías de requirements.txt: OK")

# ---- CUDA / GPU ----
def nvidia_smi():
    exe = shutil.which("nvidia-smi") or next((p for p in (r"C:\\Windows\\System32\\nvidia-smi.exe",
          r"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe") if os.path.exists(p)), None)
    if not exe:
        return None, None
    out = subprocess.run([exe], capture_output=True, text=True).stdout
    m = re.search(r"CUDA Version:\\s*(\\d+)\\.(\\d+)", out)
    gpu = subprocess.run([exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return ((int(m[1]), int(m[2])) if m else None), gpu

def wheel_tag(cu):
    for lim, tag in (((12, 8), "cu128"), ((12, 6), "cu126"), ((12, 4), "cu124"), ((11, 7), "cu118")):
        if cu >= lim:
            return tag

def probe_torch():
    code = ("import json,torch;ok=torch.cuda.is_available();err=None\\n"
            "if ok:\\n"
            "    try:\\n"
            "        a=torch.randn(64,64,device='cuda');float((a@a).sum())\\n"
            "    except Exception as e:\\n"
            "        ok=False;err=repr(e)[:150]\\n"
            "print(json.dumps(dict(v=torch.__version__,cuda=torch.version.cuda,ok=ok,err=err,"
            "name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None

drv, gpu = nvidia_smi()
print("GPU (nvidia-smi):", gpu, "| CUDA máx. del driver:", drv)
info = probe_torch()          # en un subproceso: no carga torch en este kernel
print("torch:", info)
if not (info and info["ok"]):
    if drv is None:
        raise RuntimeError("No se encontró nvidia-smi / driver NVIDIA: este equipo no tiene GPU utilizable.")
    tag = wheel_tag(drv)
    if tag is None:
        raise RuntimeError(f"El driver solo soporta CUDA {drv}; se requiere al menos 11.7. Actualiza el driver NVIDIA.")
    if info and info["cuda"] and "cu" + info["cuda"].replace(".", "") == tag:
        raise RuntimeError(f"torch {info['v']} ya es un build {tag} pero no ve la GPU: revisa el driver/reinicia el equipo.")
    if not AUTO_FIX_TORCH:
        raise RuntimeError(f"torch no ve la GPU. Instala un build {tag}: pip install torch --index-url https://download.pytorch.org/whl/{tag}")
    url = f"https://download.pytorch.org/whl/{tag}"
    print(f"Instalando torch ({tag}) compatible con el driver; puede tardar y descargar varios GB…")
    pip(["torch", "--index-url", url])
    pip(["--no-deps", "--force-reinstall", "torch", "--index-url", url])
    info = probe_torch()
    print("torch:", info)
    assert info and info["ok"], "torch sigue sin ver la GPU tras reinstalar; reinicia el kernel/equipo y revisa el driver."
print(f"CUDA OK -> torch {info['v']} (CUDA {info['cuda']}) en {info['name']}")''')

md("""## 0a-bis. Paquetes opcionales (RealMLP como modelo de evaluación externa)
Se instalan con `torch` **fijado** por constraint para no reemplazar el build CUDA. Si no se pueden instalar o usar, ese modelo se omite automáticamente (el resto del benchmark no se ve afectado). No se usa ningún modelo ni peso con licencia restringida.""")
code('''AUTO_INSTALL_OPTIONAL = True
def opt_ok(mod):
    return subprocess.run([sys.executable, "-c", mod], capture_output=True, text=True).returncode == 0
CHECKS = {"pytabkit (RealMLP)": "from pytabkit import RealMLP_TD_Classifier"}
status = {k: opt_ok(v) for k, v in CHECKS.items()}
if AUTO_INSTALL_OPTIONAL and not all(status.values()):
    cons = os.path.join(ROOT, "results", "_constraints.txt")
    os.makedirs(os.path.dirname(cons), exist_ok=True)
    open(cons, "w").write(f"torch=={info['v']}\\n")
    pip(["-r", os.path.join(ROOT, "requirements-optional.txt"), "-c", cons])
    info2 = probe_torch()
    assert info2 and info2["ok"], "La instalación opcional dañó el acceso a la GPU de torch; reinstala el build CUDA."
    status = {k: opt_ok(v) for k, v in CHECKS.items()}
for k, v in status.items():
    print(("OK    " if v else "OMITIDO"), k)''')

md("## 0b. Entorno y GPU (forzada)")
code('''import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ROOT)
from decoyfs.device import device, env_report
device(force_gpu=True)          # lanza error si torch no puede usar la GPU; cambia a False para permitir CPU
from decoyfs.device import xgb_device
assert xgb_device() == "cuda", "xgboost no puede entrenar en GPU (correría en CPU y tardaría mucho más). Actualiza xgboost (>=3.0) y el driver."
print(json.dumps(env_report(), indent=1))''')

md("""## 1. Configuración
`TIER`: `smoke` (4 datasets, minutos) · `laptop` (≈36 datasets, 5×2 CV, horas) · `full` (**58 reales + 16 sintéticos**, 5×2 CV, `max_n=15000`, estabilidad por submuestreo y **todas las extensiones**: ablaciones, baselines externos Boruta/NABFS, Block-Shadow + knockoffs, modelos afinados y RealMLP; correr en la otra PC con GPU).
Extensiones (se pueden apagar con `EXT = {}`): ver `docs/DISENO_PROTOCOLO.es.md` §Extensiones.""")
code('''from decoyfs.bench.data import TIERS, MAXN_SENS, describe
from decoyfs.bench.runner import DEFAULT_CFG, run_all
from decoyfs.bench.presets import TIER_CFG, EXT, POL_TUNED, POL_EXT, SENS_MAXN, SENS_CFG
TIER = os.environ.get("DECOYFS_TIER", "full")   # smoke | laptop | full
if TIER != "full":
    print("AVISO: las extensiones (baselines, ablaciones, modelos afinados, etc.) solo se activan en el nivel full")
CFG = {**DEFAULT_CFG, **TIER_CFG[TIER]}
OUT = os.path.join(ROOT, "results", TIER)
DATASETS = TIERS[TIER]
RUN_SENSITIVITY = TIER == "full"     # sensibilidad a max_n (limitación 6): re-ejecuta MAXN_SENS con max_n=3000 y 10000
print(CFG); print(len(DATASETS), "datasets:", DATASETS)''')

md("""## 2. Ejecución (reanudable: si se interrumpe, vuelve a ejecutar esta celda)
Muestra un **panel de progreso** (barra, % , tiempo restante y fecha/hora estimada de conclusión) que abarca la corrida principal **y** la sensibilidad a `max_n` (§2b). La estimación se calcula con los tiempos ya medidos (`timing.csv`) y el tamaño (n, p, clases) de cada dataset pendiente; los folds de la repetición 0 (más pesados) se modelan aparte. Al inicio es burda y mejora conforme avanza. Cada línea del log también muestra `[% · tiempo restante · fin]`.""")
code('''from decoyfs.bench.progress import Tracker
JOBS = [(DATASETS, OUT, CFG, "main")]
if RUN_SENSITIVITY:
    JOBS += [(MAXN_SENS, os.path.join(ROOT, "results", f"{TIER}_maxn{mx}"), {**DEFAULT_CFG, **SENS_CFG, "max_n": mx}, "sens")
             for mx in SENS_MAXN]
TRK = Tracker(JOBS)
run_all(DATASETS, OUT, CFG, tracker=TRK)''')

md("""### 2b. Sensibilidad a `max_n` (limitación 6)
Re-ejecuta un subconjunto de datasets grandes con submuestreo estratificado a `max_n` ∈ {3000, 10000} (sin extensiones, 4 modelos base) y se compara luego con la corrida completa (§3.12). Reanudable. Continúa el panel de progreso del experimento completo.""")
code('''SENS = {}
if RUN_SENSITIVITY:
    TRK.new_panel()
    for mx in SENS_MAXN:
        cfg_s = {**DEFAULT_CFG, **SENS_CFG, "max_n": mx}
        SENS[mx] = os.path.join(ROOT, "results", f"{TIER}_maxn{mx}")
        run_all(MAXN_SENS, SENS[mx], cfg_s, tracker=TRK)
else:
    print("RUN_SENSITIVITY=False: omitido")''')

md("## 3. Análisis")
code('''import pandas as pd, matplotlib.pyplot as plt
from decoyfs.bench import stats as S, plots as PL
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
res, sets, feat = S.load(OUT)
print(res.dataset.nunique(), "datasets |", len(res), "filas de resultados")''')

md("### 3.1 Tabla maestra por modelo (rango medio, Δ vs ALL, parsimonia, no-inferioridad, costo, estabilidad, FPR)")
code('''MODEL = "xgb"          # logreg | knn | xgb | mlp
tab = S.summary(res, sets, MODEL, ref="ALL", metric="auroc", delta=0.01)
print(tab.attrs)
tab.round(3)''')
code('''fig = PL.cd_diagram(tab.rank_mean, tab.attrs["cd"], title=f"{MODEL}: rangos medios (CD Nemenyi)"); plt.show()
PL.pareto(tab); plt.show()''')

md("### 3.2 ¿Beneficia más a los modelos sensibles al ruido? (Δ AUROC de cada política vs ALL, por modelo)")
code('''from decoyfs.bench.models import CORE_MODELS, TUNED_MODELS, FOUNDATION_MODELS
rows = []
for m in [m for m in CORE_MODELS if m in set(res.model)]:
    t = S.summary(res, sets, m)
    for pol in ["CTRL_VOTE_MAJ", "CTRL_BORDA_BOTH", "CVK_1SE", "BORDA_MATCH", "RAND_MATCH"]:
        if pol in t.index:
            rows.append(dict(model=m, policy=pol, d_auroc=t.loc[pol, "mean_diff"], p_holm=t.loc[pol, "p_holm"],
                             noninferior=t.loc[pol, "noninferior"], frac=t.loc[pol, "frac_feat_median"]))
pd.DataFrame(rows).pivot(index="policy", columns="model", values="d_auroc").round(4)''')

md("### 3.3 Calibración de falsos positivos (Random / Shadow) y verdad conocida (sintéticos)")
code('''PL.fpr_bars(sets); plt.show()
S.truth_table(sets).round(3)''')

md("### 3.4 Estabilidad (Nogueira) y costo de selección")
code('''st = S.stability_table(sets).groupby("policy").stability.agg(["mean", "median"]).sort_values("mean", ascending=False)
cost = sets.groupby("policy").t_select.agg(["mean", "median"])
st.join(cost, rsuffix="_t_select_s").round(3)''')

md("### 3.5 Reporte de atributos (interpretabilidad): atributos confirmados con más frecuencia")
code('''f = feat.groupby(["dataset", "name"]).agg(votes=("votes", "mean"), beat_rand=("beat_rand", "mean"),
                                          beat_shad=("beat_shad", "mean"), consensus=("consensus", "mean")).reset_index()
f.sort_values(["dataset", "votes"], ascending=[True, False]).groupby("dataset").head(5)''')

md("""## 3b. Análisis de las extensiones (limitaciones 1–6)
Los resultados de 3.1–3.5 son el **análisis primario** (preespecificado). Lo siguiente es secundario/exploratorio y se reporta como tal.""")

md("### 3.6 Baselines externos (Boruta, NABFS) y controles con dependencia (Block-Shadow, knockoffs) — limitaciones 1 y 4")
code('''POL_BASE = ["ALL", "CTRL_VOTE_MAJ", "CVK_1SE", "BORDA_MATCH", "FIXK_25", "BORUTA", "NABFS", "CTRL_BLOCK_VOTE_MAJ",
            "KO_BORDA_q10", "KO_XGB_GAIN_q10"]
BASE = {}
for m in CORE_MODELS:
    if m in set(res.model):
        BASE[m] = S.summary(res, sets, m, policies=POL_BASE, min_cov=0.0)   # solo datasets con todas las políticas
        print(m, "| datasets comunes:", BASE[m].attrs["n_datasets"])
BASE["xgb"].round(3)''')
code('''# FPR real (FDP) vs proxies, en sintéticos con verdad conocida (incluye syn13-16 con dependencia)
S.truth_table(sets).loc[[p for p in POL_BASE if p in S.truth_table(sets).index]].round(3)''')

md("### 3.7 Estadística reforzada — limitación 2: IC bootstrap sobre datasets y sensibilidad al margen δ")
code('''pv = S.pivot(res, MODEL, "auroc")
DT = S.diff_table(pv, "ALL")                    # Δ vs ALL con IC95% bootstrap (media y mediana)
display(DT.round(4).head(15))
S.delta_sensitivity(pv, "ALL", (0.0, 0.005, 0.01, 0.02)).round(4)''')

md("### 3.8 Meta-regresión exploratoria (¿cuándo ayuda la selección?) — limitación 2")
code('''MR = {}
for m in [m for m in CORE_MODELS if m in set(res.model)]:
    t, nds = S.meta_regression(res, sets, OUT, model=m, policy="CTRL_VOTE_MAJ")
    MR[m] = t
    print(m, "| datasets:", nds)
    display(t.round(4))''')

md("### 3.9 Proxies de FPR vs FDP real — limitación 2")
code("S.fpr_vs_fdp(sets).round(3)")

md("### 3.10 Ablaciones del protocolo — limitación 2")
code('''ABL = S.ablation_table(res, sets, model=MODEL, ref="CTRL_VOTE_MAJ")
ABL.round(4)''')

md("### 3.11 Sensibilidad a hiperparámetros (modelos afinados) y RealMLP — limitaciones 3 y 5")
code('''DVT = S.default_vs_tuned(res, policies=POL_TUNED)
display(DVT.round(4))
FM = {}
for m in FOUNDATION_MODELS:
    if m in set(res.model):
        FM[m] = S.summary(res, sets, m, policies=POL_EXT, min_cov=0.0)
        print(m, "| datasets:", FM[m].attrs["n_datasets"]); display(FM[m].round(3))
    else:
        print(m, "no se ejecutó (paquete no disponible)")''')

md("### 3.12 Sensibilidad a `max_n` — limitación 6")
code('''MX = {}
for mx, d in SENS.items():
    r_s, _, _ = S.load(d)
    if r_s.empty:
        continue
    MX[mx] = S.maxn_sensitivity(r_s, res, model=MODEL, policies=["ALL", "CTRL_VOTE_MAJ", "CVK_1SE", "FIXK_25", "BORDA_MATCH"])
    print("max_n =", mx); display(MX[mx].round(4))''')

md("## 4. Exportar tablas para el manuscrito")
code('''T = os.path.join(OUT, "tablas"); os.makedirs(T, exist_ok=True)
for m in res.model.unique():
    try:
        S.summary(res, sets, m).round(4).to_csv(os.path.join(T, f"resumen_{m}.csv"))
    except Exception as e:
        print("resumen", m, "omitido:", repr(e)[:100])
S.truth_table(sets).round(4).to_csv(os.path.join(T, "verdad_conocida.csv"))
S.stability_table(sets).to_csv(os.path.join(T, "estabilidad.csv"), index=False)
for m, t in BASE.items(): t.round(4).to_csv(os.path.join(T, f"baselines_{m}.csv"))
DT.round(5).to_csv(os.path.join(T, "delta_ic_bootstrap.csv"))
S.delta_sensitivity(pv, "ALL", (0.0, 0.005, 0.01, 0.02)).round(5).to_csv(os.path.join(T, "delta_sensibilidad.csv"))
for m, t in MR.items(): t.round(5).to_csv(os.path.join(T, f"meta_regresion_{m}.csv"))
S.fpr_vs_fdp(sets).round(4).to_csv(os.path.join(T, "fpr_vs_fdp.csv"))
ABL.round(5).to_csv(os.path.join(T, "ablaciones.csv"))
DVT.round(5).to_csv(os.path.join(T, "default_vs_tuned.csv"), index=False)
for m, t in FM.items(): t.round(4).to_csv(os.path.join(T, f"fundacion_{m}.csv"))
for mx, t in MX.items(): t.round(5).to_csv(os.path.join(T, f"maxn_{mx}.csv"))
print("ok ->", T)''')

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks", "DecoyFS_benchmark.ipynb"))
print("notebook ok")
