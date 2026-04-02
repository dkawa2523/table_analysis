# 05 Process Catalog

Each task emits `out.json` and `manifest.json`. The entrypoint is
`python -m tabular_analysis.cli task=<task>`.

## dataset_register
- Input: `data.dataset_path`
- Output keys: `raw_dataset_id`, `raw_schema`
- Main artifacts: `schema.json`, `preview.csv`

## preprocess
- Input: `data.raw_dataset_id` or `data.dataset_path`, `preprocess.variant`, `data.split.*`
- Output keys: `processed_dataset_id`, `preprocess_variant`, `split_hash`, `recipe_hash`
- Main artifacts: `recipe.json`, `summary.md`, `preprocess_bundle.*`, `schema.json`, `split.json`

## train_model
- Input: `train.inputs.preprocess_run_dir`, `data.processed_dataset_id`, `model_variant`, `eval.*`
- Output keys: `processed_dataset_id`, `split_hash`, `recipe_hash`, `train_task_id`, `model_id`, `best_score`, `primary_metric`, `task_type`, `n_classes`
- Main artifacts: `metrics.json`, `preds_valid.parquet`, `classes.json` for classification, `model_bundle.*`

## train_ensemble
- Input: `run.usecase_id`, `preprocess.variant`, `ensemble.method`, `ensemble.top_k`, `ensemble.selection_metric`, `ensemble.exclude_variants`
- Candidate train runs must already provide `preds_valid.parquet`; local rerun fallback is intentionally removed.
- Output keys: `processed_dataset_id`, `split_hash`, `recipe_hash`, `train_task_id`, `model_id`, `best_score`, `primary_metric`, `task_type`, `n_classes`
- Main artifacts: `metrics.json`, `ensemble_spec.json`, `model_bundle.joblib`

## leaderboard
- Input: `leaderboard.train_task_ids` or `leaderboard.train_run_dirs`
- Output keys: `leaderboard_csv`, `recommended_train_task_id`, `recommended_model_id`, `excluded_count`
- Main artifacts: `leaderboard.csv`, `recommendation.json`, `summary.md`

## infer
- Input: `infer.model_id` or `infer.train_task_id`, `infer.mode`
- Output keys: `predictions_path`

## pipeline
- Orchestrates `dataset_register -> preprocess -> train_model -> train_ensemble -> leaderboard -> infer`
- Main artifacts: `pipeline_run.json`, `report.md`, `report.json`, `report_links.json`
- ClearML controller mode is `run.clearml.execution=pipeline_controller`
