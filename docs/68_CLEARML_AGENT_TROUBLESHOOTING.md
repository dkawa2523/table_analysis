# ClearML Agent Troubleshooting

## Quick triage order
1. Check the pipeline task script.
2. Check queue and agent status.
3. Check agent-side dependency errors.

## 1) Pipeline task script
- `repository` and `branch` must point to the solution repo, not the platform repo.
- `entry_point` must be `tools/clearml_entrypoint.py`.

```bash
python - <<'PY'
from clearml import Task

task = Task.get_task(task_id="<PIPELINE_TASK_ID>")
print(task.get_script())
PY
```

If the script is wrong, refresh templates and rerun the controller task:

```bash
python tools/clearml_templates/manage_templates.py --apply --project-root LOCAL
python tools/clearml_templates/manage_templates.py --validate --project-root LOCAL
```

## 2) Queue / agent basics
- `run.clearml.queue_name` is not the canonical child-routing knob for pipeline runs.
- In pipeline mode, `exec_policy.queues.*` is the source of truth for child task queues.
- Use `run.clearml.queue_name` only when you intentionally want to steer the controller task itself.
- The agent must appear online in the ClearML UI.
- If tasks stay queued, verify that the queue has at least one healthy worker.
- Canonical container assets live under `tools/clearml_agent/`.
- Use a Docker named volume for `/root/.clearml` and set `UV_CACHE_DIR=/root/.clearml/uv-cache`.
- Avoid Windows bind mounts for `/root/.clearml`. They can push agent processes into `p9_client_rpc` waits and stall controller runs.
- The canonical agent image already starts through `tini`; do not reintroduce Windows host bind mounts just to share `/root/.clearml`.
- Canonical queue split is:
  - `controller`: pipeline controllers only
  - `default`: preprocess, light train models, leaderboard, ensembles
  - `heavy-model`: `catboost` and `xgboost`

## 3) Dependency and bootstrap errors
- `ModuleNotFoundError` usually means the repo bootstrap or optional dependency set is incomplete.
- Confirm the task uses `tools/clearml_entrypoint.py`.
- Confirm the environment includes `-e .` or the equivalent editable install path.
- Recheck optional extras when using serving, Optuna, or non-core models.
- Current bootstrap policy is task-specific:
  - `dataset_register`, `preprocess`, `leaderboard`, `pipeline`: base dependencies only
  - `train_model`: only the optional extra needed by the selected model
  - `lgbm`: `lightgbm` only
  - `xgboost`: `xgboost` only
  - `catboost`: `catboost` only
  - `infer`: model-specific extra when known, otherwise `models` fallback
  - `infer.mode=optimize`: add `optuna`

## Hydra list override pitfalls
Bad:
`pipeline.selection.enabled_model_variants=["catboost", "elasticnet"]`

Good:
`pipeline.selection.enabled_model_variants=[catboost,elasticnet]`

Use the same Hydra list style for preprocess variants and other list overrides.

## Project override rule
- Prefer task-level `task.project_name`.
- Avoid ad hoc `run.clearml.project_name` overrides in child tasks unless the config explicitly allows them.

