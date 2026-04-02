# 65 Dev Guide Directory Map

## Purpose
This file is the quick map for engineers who need to find the current source of
truth without reading the whole repository first.

## Quick task flow
- Runtime work usually follows `dataset_register -> preprocess -> train_model`.
- Orchestration work usually enters at `pipeline.py` and ends in leaderboard or infer.
- For task-by-task behavior, start with `docs/05_PROCESS_CATALOG.md` and then jump to the matching process file.

## Runtime entry points
- `src/tabular_analysis/cli.py`
  - Hydra CLI entry point.
- `src/tabular_analysis/processes/`
  - Task implementations: `dataset_register.py`, `preprocess.py`, `train_model.py`, `train_ensemble.py`, `leaderboard.py`, `infer.py`, `pipeline.py`, `retrain.py`.
- `src/tabular_analysis/processes/lifecycle.py`
  - Shared runtime bootstrapping and `out.json` / `manifest.json` emission.

## Config layout
- `conf/task/`
- `conf/group/model/`
- `conf/group/preprocess/`
- `conf/pipeline/model_sets/`
- `conf/clearml/`

## Variant and preprocessing logic
- `src/tabular_analysis/registry/models.py`
- `src/tabular_analysis/processes/preprocess.py`
- `src/tabular_analysis/feature_engineering/`
- `src/tabular_analysis/common/feature_types.py`

## ClearML integration
- `src/tabular_analysis/platform_adapter_core.py`
- `src/tabular_analysis/platform_adapter_task.py`
- `src/tabular_analysis/platform_adapter_artifacts.py`
- `src/tabular_analysis/platform_adapter_model.py`
- `src/tabular_analysis/platform_adapter_pipeline.py`
- `src/tabular_analysis/platform_adapter_clearml_env.py`
- `src/tabular_analysis/clearml/`
- `src/tabular_analysis/ops/clearml_identity.py`

## Serving
- `src/tabular_analysis/serve/`
- `docs/22_SERVING.md`

## Useful companion docs
- `docs/03_CLEARML_UI_CONTRACT.md`
- `docs/05_PROCESS_CATALOG.md`
- `docs/10_OPERATION_MODES.md`
- `docs/67_REHEARSAL_COMMANDS.md`
- `docs/69_CLEARML_TROUBLESHOOTING.md`
- `docs/81_CLEARML_TEMPLATE_POLICY.md`
- `docs/84_REHEARSAL_GUIDE.md`
