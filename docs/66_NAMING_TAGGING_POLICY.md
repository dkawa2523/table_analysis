# 66 Naming Tagging Policy

## Goal
Task names, tags, and properties should be predictable enough for operators and
automation to find the right ClearML objects quickly.

## Task names
- Respect the process name first: `dataset_register`, `preprocess`, `train_model`, `leaderboard`, `infer`, `pipeline`, `retrain`
- Variant-specific naming is resolved in the shared naming helpers
- `run.clearml.task_name` may override the visible task name, but should not replace process identity

## Required tags
Tags are built by the shared ClearML helpers:
- `usecase:<usecase_id>`
- `process:<process>`
- `schema:<schema_version>`
- `grid:<grid_run_id>` when present
- `retrain:<retrain_run_id>` when present

## Optional tags
- `run.clearml.policy.tags`
- `run.clearml.policy.extra_tags`
- `run.clearml.extra_tags`
- process-specific tags such as `preprocess:<variant>`, `grid_cell:<preprocess>__<model>`, `hpo:<trial>`

## User properties
Base properties are built by the shared ClearML helpers:
- `usecase_id`
- `process`
- `schema_version`
- `code_version`
- `platform_version`
- `grid_run_id`
- `retrain_run_id` when present

Process-specific properties may add task outputs such as `processed_dataset_id`,
`split_hash`, `model_id`, `primary_metric`, or `decision`.
