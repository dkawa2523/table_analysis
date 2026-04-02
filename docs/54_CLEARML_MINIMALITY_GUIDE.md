# 54 ClearML Minimality Guide

## Goal
Keep ClearML behavior understandable by avoiding duplicated policy and scattered
API calls.

## Rules
1. Keep ClearML API calls inside the adapter family modules and `src/tabular_analysis/clearml/`.
2. Keep process files focused on orchestration and domain logic.
3. Prefer one canonical config path over compatibility aliases.
4. Prefer explicit task/artifact contracts over runtime fallback behavior.

## Main ownership
- `platform_adapter_*`: task, artifact, model, pipeline, and environment helpers
- `clearml/hparams.py`: HyperParameters sections
- `clearml/templates.py`: template lookup
- `clearml/ui_logger.py`: UI logging helpers
