# DecoyFS — Protocolo de selección de atributos referenciado a controles (Random + Shadow)

> Nombre de trabajo (cámbialo cuando quieras). Proyecto **independiente** del proyecto de Mónica Jauregui; nace de las lecciones del framework de sus Experimentos 1 y 2.
> Código: `src/decoyfs/` · CLI: `benchmarks/run_benchmark.py` · Notebook: `benchmarks/build_notebook.py` (genera `DecoyFS_benchmark.ipynb`).

## 1. Idea central

Todo método de selección entrega un *ranking* o *score*, pero casi nunca dice **dónde cortar** ni **si el atributo tiene señal real**. El protocolo añade dos controles negativos a cada fold, calculados solo con *train*:

| Control | Qué es | Qué detecta |
|---|---|---|
| **Random probes** | 7 familias (uniforme, normal, lognormal, exponencial, beta dispersa, Bernoulli, Poisson) | artefactos de escala/cardinalidad/ceros (p. ej. ReliefF y MI premiando a un Random en el Exp. 1) |
| **Shadow** | copia de cada atributo real con filas permutadas (misma marginal, sin relación con y) | si un atributo no supera a sus propias sombras, no aporta señal sobre la clase (tipo Boruta) |

Un atributo se **confirma** cuando supera al máximo de los Random **y** al cuantil 95 % de sus Shadow. Con esto:
1. el **punto de corte es automático** (sin tunear k),
2. se obtiene una **tasa de falsos positivos estimada** para *cualquier* política de selección (fracción de controles que "pasarían" el corte),
3. los métodos y políticas pueden compararse en un terreno común.

## 2. Preguntas e hipótesis (con criterio de éxito)

| # | Hipótesis | Cómo se mide | Criterio |
|---|---|---|---|
| H1 | **No-inferioridad**: el conjunto confirmado rinde igual que usar todos los atributos | ΔAUROC vs `ALL` por dataset; Wilcoxon unilateral con margen δ = 0.01 + Holm | p_noninf < 0.05 |
| H2 | **Mejora en modelos sensibles al ruido** (kNN, MLP, LogReg) y en p ≫ n / datasets con ruido | ΔAUROC vs `ALL` por modelo; metarregresión vs p/n y % de ruido | Δ > 0 significativo; en GBDT se espera ≈ 0 (resultado honesto) |
| H3 | **Control de falsos positivos**: menor FPR/FDP que top-k fijo o 1 solo método | `fpr_rand`, `fpr_shad` (todos los datasets); FDP y recall con verdad conocida (sintéticos) | FDP menor a igual recall |
| H4 | **A igual tamaño**, el conjunto confirmado supera a Borda-top-k, a un solo método y a un subconjunto aleatorio | políticas `*_MATCH` (mismo k que la política de referencia) | rango medio menor |
| H5 | **Estabilidad** mayor entre folds/repeticiones | índice de Nogueira (2018) | mayor que top-k/1 método |
| H6 | **Costo**: reemplaza la búsqueda de k por CV (`CVK_1SE`) con menos cómputo, y reduce entrenamiento/inferencia | `t_select`, `t_fit`, `t_pred`, k | menor tiempo total con rendimiento no inferior |
| H7 | **Validez del control negativo**: los atributos *rechazados* rinden mucho peor que los confirmados | política `REJECTED` vs referencia | AUROC(REJECTED) ≪ AUROC(ref) (nota: en datos muy redundantes, p. ej. wdbc, los rechazados aún conservan señal; se reporta) |

> Resultado negativo aceptable y publicable: "en GBDT la selección no mejora la exactitud, pero reduce 60–90 % de atributos sin pérdida significativa y con FDP calibrado".

## 3. Datasets

* **Sintéticos con verdad conocida (16)** – `syn01…syn12` (independientes) y `syn13…syn16` (**con dependencia**: bloques correlacionados, p≫n con ρ=0.8, XOR sin señal marginal, proxies correlacionados con causales). Permiten medir FDP/recall reales.
* **Reales públicos (OpenML, IDs verificados): 58** – regímenes:
  * *Tabulares clásicos* (p 5–72): wdbc, diabetes, credit-g, ilpd, kc1, pc4, climate, steel-plates, ozone, spambase, qsar-biodeg, vehicle, segment, satimage, optdigits, texture, magic, eeg-eye, wall-robot.
  * *Con ruido conocido (AutoML/NIPS-2003)*: madelon, philippine, madeline, sylvine, jasmine, christine, gina.
  * *Alta dimensión*: cnae-9, har, isolet, bioresponse, arcene, internet-ads, micro-mass, mfeat-factors, dna.
  * *Grandes en n*: bank, nomao, electricity (submuestreo estratificado con `max_n`).
  * *Extensión de validez externa (20, de OpenML-CC18; sin duplicar los anteriores)*: vowel, adult, letter, pendigits, eucalyptus, churn, jm1, pc1, sick, phishing, gesture, kr-vs-kp, cylinder-bands, mfeat-zernike, first-order-tp, splice, authorship, mice-protein, semeion, mfeat-fourier.
* Ampliación recomendada para el artículo: dominio celular/biomédico (JUMP Cell Painting o BBBC con atributos de CellProfiler) como caso "tipo CTC" público.

**Tiers:** `smoke` (4 datasets) · `laptop` (24 reales + 12 sintéticos) · `full` (58 reales + 16 sintéticos = 74, con extensiones §12).

## 4. Protocolo (por dataset)

```
CV externa estratificada 5×R  (R=2 en laptop y en full)
 └─ por fold (todo con TRAIN únicamente):
     1. FoldPrep: imputación mediana → quita constantes → Yeo-Johnson (|skew|>1, >10 valores) → escala robusta → clip ±10
     2. Aumentar: [reales | Random (≥14) | Shadow (≥40 copias)]
     3. 7 rankers (ReliefF/χ²/MI sobre cuantiles uniformes): ANOVA-F, χ²q, MI-bin, ReliefF, RF-MDI, XGB-gain, L1-LogReg
     4. Políticas de selección (ver §5) → conjuntos de atributos
     5. Modelos sobre cada conjunto único: LogReg, kNN(GPU), XGBoost(GPU), MLP(GPU)
     6. Evaluación en el fold de prueba: AUROC (macro OvR), PR-AUC, bal-acc, macro-F1, logloss, k, tiempos
```
Sin fuga: selección, controles y preprocesamiento se ajustan solo con train; el fold de prueba nunca se usa para decidir. Desbalance tratado igual en todos (class weights) y no es el objeto de estudio.

## 5. Políticas comparadas

| Grupo | Políticas |
|---|---|
| Referencia | `ALL` |
| Convencionales | `FIXK_10/25/50` (% de p por consenso Borda) · `CVK_1SE` (k por CV interna + regla 1-ES) |
| **Referenciadas a controles** | `CTRL_BORDA_RAND`, `CTRL_BORDA_SHAD`, `CTRL_BORDA_BOTH` (corte en Borda) · `CTRL_VOTE_ANY/MAJ/ALL` (nº de métodos que confirman) · `CTRL_STAB` (frecuencia ≥ 0.6 en B submuestreos; `stab_B>0`) |
| Controles de ablación (mismo k = política de referencia `CTRL_VOTE_MAJ`) | `BORDA_MATCH`, `TOPK_MATCH_<método>`, `RAND_MATCH` (null), `REJECTED` (control negativo) |

Si una política devuelve el conjunto vacío, se usa el top-3 por consenso y se marca `fallback=True`.

## 6. Estadística

* **Entre datasets (inferencia principal):** Friedman + rangos medios + CD de Nemenyi; Wilcoxon pareado + Holm vs `ALL`; **no-inferioridad** (δ=0.01 AUROC); resultados por modelo y por régimen (sintético / p≪n / p≫n / ruido conocido).
* **Dentro de dataset:** t corregida de Nadeau–Bengio (función `nadeau_bengio`).
* **Estabilidad:** Nogueira sobre las 5×R selecciones. **Calibración:** FPR estimada con Random/Shadow en todos; FDP/recall exactos en sintéticos.
* **Metaanálisis (exploratorio):** ΔAUROC ~ log n + log p + p/n + minoría + fracción de ruido estimada (modelo mixto con intercepto aleatorio por dataset) → *cuándo* conviene seleccionar (§12).
* **Ablaciones del propio método (tier full, §12):** α ∈ {0.01, 0.10}, solo-Random, solo-Shadow, mitad de las sondas Random, leave-one-ranker-out. (No se ablaciona el nº de réplicas de Shadow: con p ≥ 40 solo hay 1 copia.) `stab_B` se fija en 5.

## 7. Cómputo y GPU

**Estado en esta laptop (NVIDIA T500, 2 GB, driver 517.89 = CUDA 11.7):** el torch global (cu124) **no** ve la GPU. Solución aplicada y verificada: venv `.venv` con `torch 2.6.0+cu118` → `torch.cuda.is_available() == True` y xgboost entrena en `cuda`. Kernel de Jupyter: **"DecoyFS (.venv, GPU cu118)"**.

```powershell
cd ...\FS_Framework
.\.venv\Scripts\python.exe run_benchmark.py --tier smoke  --force-gpu
.\.venv\Scripts\python.exe run_benchmark.py --tier laptop --force-gpu
```
`--force-gpu` aborta si la GPU no funciona (evita correr 10 h en CPU sin saberlo). Alternativa más limpia: actualizar el driver NVIDIA (≥ 528) y usar el Python global.

GPU se usa en: ANOVA, χ², MI, **ReliefF** (cdist L1), kNN, MLP y XGBoost. RF, L1-LogReg y preprocesamiento son CPU. Con 2 GB de VRAM los kernels están troceados; el beneficio de la GPU es real en ReliefF/kNN/MLP con n grande, pero modesto en datasets chicos (el tiempo lo domina el entrenamiento de XGB: ~2 s por ajuste).

**Tiempos medidos (smoke, GPU):** 10–25 s por fold en datasets de p≈30–50 y 25–125 s en madelon (p=500). Estimación `laptop`: ~360 folds × 30–120 s ≈ **6–12 h** (reanudable). `full` conviene en la otra PC.

## 8. Resultados preliminares (smoke, 4 datasets — solo para validar el pipeline, NO concluyentes)

* Sintéticos: `CTRL_VOTE_MAJ` → FDP ≈ 1.7 % con recall de señal ≈ 0.90; top-k fijo 25–50 % → FDP 26–60 %; subconjunto aleatorio del mismo tamaño → AUROC 0.88 vs 0.97.
* FPR estimada (Random/Shadow) de `CTRL_VOTE_MAJ` ≈ 0.10; de `FIXK_50` ≈ 0.32–0.45; de `ALL` ≈ 0.9–1.0.
* kNN: +0.05 AUROC vs `ALL` con ~43 % de los atributos; XGB: no inferior (≈ sin cambio). Consistente con H2.

## 9. Amenazas a la validez

1. Con pocos datasets Friedman/Nemenyi tiene poca potencia → el nivel full usa 74 (58 reales).
2. Hiperparámetros fijos en el análisis principal; se mitiga con un subconjunto con modelos afinados (§12.3).
3. Shadow por permutación subestima el nulo con atributos correlacionados → se complementa con Block-Shadow y knockoffs gaussianos (§12.2) y con la verdad conocida sintética.
4. Datasets de OpenML pueden estar ya preprocesados/duplicados; reportar duplicados y fuentes.
5. `max_n` submuestrea datasets grandes → sensibilidad con `max_n` ∈ {3000, 10000} frente a la corrida de referencia con `max_n`=15000 (§12.6).
6. Codificación categórica (one-hot ≤12 niveles) se decide con el diccionario de niveles completo (no usa y).

## 10. Aportaciones esperadas (ver también el resumen en el chat)

1. **Protocolo** con corte automático y FPR estimada para cualquier selector (nadie pide k).
2. **Benchmark** con verdad conocida + datasets reales bajo un mismo protocolo sin fuga, con estadística correcta entre datasets (Demšar) y no-inferioridad.
3. **Evidencia de cuándo la selección ayuda**: por modelo, régimen (p/n) y nivel de ruido, incluidos resultados negativos para GBDT.
4. **Diagnóstico de artefactos de escala** (ReliefF/MI/RF-MDI frente a Random) cuantificado en muchos datasets.
5. **Software abierto** reproducible (GPU/CPU, reanudable) y reporte de atributos por fold (interpretabilidad).

## 11. Plan

| Fase | Contenido |
|---|---|
| 0 ✔ | Infraestructura, GPU, smoke test |
| 1 | `laptop` completo (24 reales + 12 sintéticos) → análisis preliminar |
| 2 | Ablaciones del método (α, nº probes, leave-one-ranker-out, `stab_B`) |
| 3 | `full` en la otra PC (alta dimensión, grandes en n) + dominio celular público |
| 4 | Metaanálisis, figuras, revisión sistemática formal de literatura, manuscrito |

## 12. Extensiones (limitaciones 1–6; activas en el nivel `full`, ver `decoyfs/bench/presets.py`)

| # | Limitación | Qué se añadió | Dónde |
|---|---|---|---|
| 1 | Faltan baselines con ruido | `BORUTA` (estilo Boruta: XGBRF en GPU, aciertos vs máx. Shadow, binomial + Bonferroni, presupuesto de 240 s) y `NABFS` (reimplementación de Sinha et al.: ruido gaussiano, bootstrap, Wilcoxon + Holm). **Son reimplementaciones propias, no los códigos originales.** | `decoyfs/comparators.py` |
| 2 | Estadística | IC bootstrap sobre datasets (media y mediana), sensibilidad al margen δ ∈ {0, .005, .01, .02}, meta-regresión mixta (exploratoria), proxies de FPR vs FDP real, ablaciones (`ABL_*`) | `decoyfs/bench/stats.py`, `policies.ablation_sets` |
| 3 | Hiperparámetros fijos | Modelos afinados (`logreg_t, knn_t, xgb_t, mlp_t`; holdout interno 25 %, cuadrículas pequeñas) en un subconjunto de políticas, solo rep 0 y p ≤ 1000 | `models.Tuned`, `presets.EXT.model_plan` |
| 4 | Shadow ignora dependencia | `CTRL_BLOCK_VOTE_MAJ` (Block-Shadow: una permutación por cluster |ρ|>0.5), `KO_BORDA_q10` / `KO_XGB_GAIN_q10` (knockoffs gaussianos model-X, Σ Ledoit–Wolf, knockoff+ q=0.10; se omiten si p > 2000), columnas `fpr_block`, `fpr_ko`; sintéticos `syn13–16` | `decoyfs/controls.py`, `runner._pass2` |
| 5 | Validez externa | +20 datasets OpenML-CC18; RealMLP-TD (pytabkit) como modelo adicional (rep 0, p ≤ 500, n ≤ 10 000; TabPFN se descartó porque sus pesos tienen licencia no comercial y exigen token, lo que impide una réplica abierta) | `models.RealMLP` |
| 6 | `max_n` | Re-ejecución de `MAXN_SENS` (bank, nomao, electricity, magic, letter, eeg-eye, adult, gina, jm1) con `max_n` ∈ {3000, 10000}, comparada con la corrida de referencia (`max_n`=15000; en el nivel `full` solo 6 datasets superan ese tope: adult, electricity, bank, nomao, letter, magic) | notebook §2b/§3.12, `--sensitivity` |

**Costo adicional:** la 2ª pasada (Block-Shadow + knockoffs) duplica aprox. el costo de rankers; Boruta/NABFS añaden de segundos a minutos por fold según p. El nivel `full` (5×2, `max_n`=15000, con extensiones) es una corrida de **varias decenas de horas a pocos días** en una GPU de gama media; reanudable. Apagar extensiones: `run_benchmark.py --no-ext`.

**Estatus:** el análisis primario (H1–H7, §6) es preespecificado; todo lo de §12 es secundario/exploratorio y se reporta como tal.
