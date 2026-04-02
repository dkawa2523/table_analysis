# 24_REPORTING (Pipeline Summary Outputs)

This document describes the pipeline report artifacts produced at the end of a pipeline run.

## Outputs
- `report.md` (human readable, one-page summary for non-DS users)
- `report.json` (machine readable summary)
- `report_links.json` (run_dir and ClearML URLs for each task, when available)

## report.md content (minimum)
- Conclusion: `grid_run_id`, recommended model, primary metric, best score, status
- Data overview: dataset/schema summary and dataset IDs
- Comparability: split/recipe hashes, processed_dataset_id, primary metric, direction
- Top models table (Top N)
- Recommendation rationale + threshold (when available)
- Notes: imbalance, calibration, uncertainty status

## report.json structure (summary)
`report.json` mirrors the same information in structured form:
- `report_version`
- `grid_run_id`
- `status`
- `summary` (recommended_model_id, primary_metric, best_score, models_tried, planned/executed jobs, rationale)
- `dataset` (raw_dataset_id, processed_dataset_id, rows, feature_columns, target_column, id/drop columns)
- `data_quality` (summary when available)
- `split` (preprocess_variant, split_hash, recipe_hash, split settings)
- `comparability` (require_comparable, split_hash, recipe_hash, direction, task_type)
- `top_models` (rank, model_variant, preprocess_variant, best_score, metric)
- `recommendation` (model_id, train_task_ref, primary_metric, direction, thresholding/calibration/imbalance/uncertainty)
- `notes` (capabilities status)
- `next_actions` (operator guidance)

## report_links.json
`report_links.json` provides a quick navigation index:
- `pipeline`, `dataset_register` (optional), `preprocess` (list), `train` (list), `leaderboard`, `infer`
- `dataset_register` は pipeline が dataset_register を含めないため null になる場合がある。
- Each entry includes `run_dir`, optional `task_id`, and `clearml_url` when ClearML is enabled.

## ClearML integration (optional)
When ClearML is enabled and reporting APIs are available, the pipeline report markdown is also published to
ClearML (in addition to being uploaded as artifacts). If ClearML is disabled or reporting is unavailable, the
behavior is a no-op and only local artifacts are produced.
