# Changelog

## 0.1.0

- First public release: `DecoySelector` (Random + Shadow decoys, 7 built-in rankers, `all` / `majority` / `any` / integer / callable vote policies, optional stability filter and Block-Shadow decoys).
- Comparator selectors: `FixedKSelector`, `CVKSelector`, `BorutaSelector` (Boruta-style re-implementation, not `BorutaPy`), `NABFSSelector`; `compare_selectors` helper.
- Registries to add or remove rankers and policies (`register_ranker`, `register_policy`, ...).
- Input validation with actionable errors (`DecoyInputError`).
- `decoyfs.bench`: the benchmark harness behind the paper (74 datasets, 5x2 CV, 4 downstream models, paired statistics).
