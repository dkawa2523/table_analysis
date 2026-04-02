# 53 ClearML HyperParameters Contract

## Goal
ClearML HyperParameters should expose a compact, sectioned view of runtime
inputs without dumping the full resolved config into the UI.

## Current contract
- `src/tabular_analysis/clearml/hparams.py` extracts dotpaths into named sections
- `src/tabular_analysis/platform_adapter_task.py` connects those sections through `Task.connect`
- `config_resolved.yaml` remains the full artifact for complete inspection

## Expected sections
- `inputs`
- `dataset`
- `preprocess`
- `model`
- `eval`
- `pipeline`
- `clearml`

Detailed section mapping lives in `docs/61_CLEARML_HPARAMS_SECTIONS.md`.
