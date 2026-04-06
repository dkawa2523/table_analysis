# ClearML Agent Troubleshooting

## Recommended baseline for a new repo / new PC / new ClearML server

When this solution moves to a new Git repository, a new PC, or a different
ClearML server, align the agent environment first.

Recommended values:

- `CLEARML_API_HOST=http://<host>:8008`
- `CLEARML_WEB_HOST=http://<host>:8080`
- `CLEARML_FILES_HOST=http://<host>:8081`
- `CLEARML_API_ACCESS_KEY=<access-key>`
- `CLEARML_API_SECRET_KEY=<secret-key>`
- Docker named volume for `/root/.clearml`
- `UV_CACHE_DIR=/root/.clearml/uv-cache`

Canonical assets:

- [compose.yaml](d:/tabular_clearml/ml_taularanalysis_v1-master/tools/clearml_agent/compose.yaml)
- [tools/clearml_agent/.env.example](d:/tabular_clearml/ml_taularanalysis_v1-master/tools/clearml_agent/.env.example)

Bring-up order:

1. Create the Docker network if needed
2. Copy `.env.example` to `.env`
3. Fill in the ClearML server endpoints and credentials
4. Start the `controller`, `default`, and `heavy-model` agents
5. Run `manage_templates --apply` and `--validate`
6. Verify that seed task scripts point to the new solution repository

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

If the solution repo moved to a new Git account or a new repository, rerun the
template sync so task scripts point to the new repository.

## 2) Queue / agent basics

- `run.clearml.queue_name` is not the canonical child-routing knob for pipeline runs.
- In pipeline mode, `exec_policy.queues.*` is the source of truth for child task queues.
- Use `run.clearml.queue_name` only when you intentionally want to steer the controller task itself.
- The agent must appear online in the ClearML UI.
- If tasks stay queued, verify that the queue has at least one healthy worker.
- Canonical container assets live under `tools/clearml_agent/`.
- Copy `tools/clearml_agent/.env.example` to `.env` as the default starting point.
- Use a Docker named volume for `/root/.clearml` and set `UV_CACHE_DIR=/root/.clearml/uv-cache`.
- Avoid Windows bind mounts for `/root/.clearml`. They can push agent processes into `p9_client_rpc` waits and stall controller runs.
- The canonical agent image already starts through `tini`; do not reintroduce Windows host bind mounts just to share `/root/.clearml`.
- Canonical queue split is:
  - `controller`: pipeline controllers only
  - `default`: preprocess, light train models, leaderboard, ensembles
  - `heavy-model`: `catboost` and `xgboost`

If you rename queues for your environment, update the queue names consistently in:

- agent `compose.yaml`
- execution policy config
- any operator-facing docs or runbooks

## 3) Dependency and bootstrap errors

- `ModuleNotFoundError` usually means the repo bootstrap or optional dependency set is incomplete.
- Confirm the task uses `tools/clearml_entrypoint.py`.
- Confirm the environment includes `-e .` or the equivalent editable install path.
- Recheck optional extras when using serving, Optuna, or non-core models.
- If `ml_platform` moved to a different Git repository, update the dependency URL in the solution repo before rerunning remote tasks.
- If either repository is private, confirm that the agent container can clone both the solution repo and the `ml_platform` dependency repo.

Current bootstrap policy is task-specific:

- `dataset_register`, `preprocess`, `leaderboard`, `pipeline`: base dependencies only
- `train_model`: only the optional extra needed by the selected model
- `lgbm`: `lightgbm` only
- `xgboost`: `xgboost` only
- `catboost`: `catboost` only
- `infer`: model-specific extra when known, otherwise `models` fallback
- `infer.mode=optimize`: add `optuna`

## 4) Private Git repository pitfalls

If the solution repo or `ml_platform` repo is private, remote tasks can fail even
when local setup works.

Check:

- the repository URL embedded in the task script
- the Git URL embedded in the dependency declaration
- whether the agent container has read access to both repos

Recommended order:

1. First validate with public or internal read-only access if possible
2. Then add private access for the agent container
3. If private Git access is difficult, distribute `ml_platform` as a package or wheel

## Hydra list override pitfalls

Bad:

`pipeline.selection.enabled_model_variants=["catboost", "elasticnet"]`

Good:

`pipeline.selection.enabled_model_variants=[catboost,elasticnet]`

Use the same Hydra list style for preprocess variants and other list overrides.

## Project override rule

- Prefer task-level `task.project_name`.
- Avoid ad hoc `run.clearml.project_name` overrides in child tasks unless the config explicitly allows them.
