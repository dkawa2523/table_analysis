# 13 Platform Integration

The solution keeps platform-facing behavior inside the adapter family modules
under `src/tabular_analysis/`.

## Canonical modules
- `platform_adapter_core.py`: shared ClearML/script/version helpers
- `platform_adapter_task.py`: task lifecycle, properties, tags, task ops
- `platform_adapter_artifacts.py`: artifact and manifest helpers
- `platform_adapter_model.py`: model registry/reference helpers
- `platform_adapter_pipeline.py`: PipelineController helpers
- `platform_adapter_clearml_env.py`: environment-aware ClearML helpers

## Why this split exists
- Process code should import only the helper family it needs.
- ClearML/script/version policy should stay centralized.
- Artifact writing should stay consistent across tasks.

## Current note
- Dataset registration still uses direct ClearML dataset APIs where `ml_platform`
  does not yet provide the needed abstraction.
