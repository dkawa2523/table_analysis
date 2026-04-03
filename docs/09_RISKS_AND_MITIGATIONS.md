# 09 Risks And Mitigations

This document tracks the main structural risks in the solution repository and
the mitigation used to keep implementation and operations understandable.

## 1. Platform API drift
**Mitigation**
- Keep solution-side platform calls behind the adapter family modules under `src/tabular_analysis/`.
- Pin compatible platform versions when needed.
- Prefer adapter changes over process-wide call-site changes.

## 2. ClearML UI drift
**Mitigation**
- Keep the UI contract documented in `docs/03_CLEARML_UI_CONTRACT.md`.
- Route naming, tags, and HyperParameters behavior through the shared ClearML modules.

## 3. Comparability regressions in leaderboard selection
**Mitigation**
- Keep split and recipe identity emitted from preprocess.
- Require `processed_dataset_id`, `split_hash`, and `recipe_hash` where model comparisons matter.

## 4. Grid complexity growing too quickly
**Mitigation**
- Keep model-set expansion centralized in pipeline config and model registry.
- Keep execution limits and failure policy explicit in config and reports.

## 5. Train / infer contract mismatch
**Mitigation**
- Keep preprocessing inside the bundle boundary.
- Use canonical model references for `registry_model_id`, local bundle path, and train task artifact resolution.

## 6. Historical docs and generated residue drift
**Mitigation**
- Keep canonical operational entry points limited to the current CLI, template tool, and rehearsal runner.
- Clean generated residue with `tools/cleanup_repo.py`.
- Remove stale docs and wrappers in the same change instead of leaving historical alternatives behind.

