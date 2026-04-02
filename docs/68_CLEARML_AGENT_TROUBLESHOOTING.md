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
- `run.clearml.queue_name` must match the queue the agent is watching.
- The agent must appear online in the ClearML UI.
- If tasks stay queued, verify that the queue has at least one healthy worker.

## 3) Dependency and bootstrap errors
- `ModuleNotFoundError` usually means the repo bootstrap or optional dependency set is incomplete.
- Confirm the task uses `tools/clearml_entrypoint.py`.
- Confirm the environment includes `-e .` or the equivalent editable install path.
- Recheck optional extras when using serving, Optuna, or non-core models.

## Hydra list override pitfalls
Bad:
`pipeline.grid.model_variants=["catboost", "elasticnet"]`

Good:
`pipeline.grid.model_variants=[catboost,elasticnet]`

Use the same Hydra list style for preprocess variants and other list overrides.

## Project override rule
- Prefer task-level `task.project_name`.
- Avoid ad hoc `run.clearml.project_name` overrides in child tasks unless the config explicitly allows them.
