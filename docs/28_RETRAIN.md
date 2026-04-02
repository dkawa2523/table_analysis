# 28_RETRAIN (Monitoring -> Retrain -> Recommend)

This document describes the retrain orchestration that connects monitoring signals to a full retrain loop.

## Goal
- Re-run the training pipeline on new data and capture the recommendation.
- Provide a reproducible decision artifact for user selection at inference time.

## Inputs
- Latest data:
  - `data.dataset_path` (local CSV/Parquet), or
  - `data.raw_dataset_id` (ClearML Dataset ID)

## Flow
1) Run `pipeline` with the latest dataset (exec_policy is respected).
2) Read the leaderboard recommendation (challenger).
3) Emit `retrain_decision.json` for user selection.

## Outputs
Retrain task artifacts:
- `retrain_summary.md`
- `retrain_decision.json` (recommended model metadata)
- `retrain_run.json` (references to pipeline/leaderboard)

Pipeline/compare outputs remain in their own stages:
- `99_pipeline/pipeline_run.json`
- `05_leaderboard/decision_summary.json` (via pipeline)

## ClearML Traceability
- Tasks carry `grid:<grid_run_id>` and `retrain:<retrain_run_id>` tags.
- `retrain_run_id` is stored in user properties for the retrain task.

## Examples
Local retrain:
```bash
python -m tabular_analysis.cli task=retrain \
  run.clearml.enabled=false \
  run.output_dir=outputs/20260102_090000 \
  data.dataset_path=/path/to/latest.csv \
  data.target_column=target
```
