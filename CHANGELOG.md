# Changelog

## 0.1.1

- New `DecoySelector.fdp_estimate_`: target-decoy estimates of the false-discovery proportion of the selected set (`min(1, FPR * (p - |S|) / |S|)`), from the Shadow null-pass rate (`"shadow"`) and from the fraction of Shadow copies that the vote policy would keep (`"vote"`). They are empirical estimates without a coverage guarantee; in a simulation study of 1,200 scenarios the vote-based one reached Spearman 0.58 and a mean absolute error of 0.19 against the true FDP.
- New functions `decoyfs.policies.estimate_fdp` and `decoyfs.policies.vote_null_pass`.
- New `benchmarks/calibration_study.py`: the scenario generator and runner of that study.

## 0.1.0

- First public release: `DecoySelector` (Random + Shadow decoys, 7 built-in rankers, `all` / `majority` / `any` / integer / callable vote policies, optional stability filter and Block-Shadow decoys).
- Comparator selectors: `FixedKSelector`, `CVKSelector`, `BorutaSelector` (Boruta-style re-implementation, not `BorutaPy`), `NABFSSelector`; `compare_selectors` helper.
- Registries to add or remove rankers and policies (`register_ranker`, `register_policy`, ...).
- Input validation with actionable errors (`DecoyInputError`).
- `decoyfs.bench`: the benchmark harness behind the paper (74 datasets, 5x2 CV, 4 downstream models, paired statistics).
