# 81 ClearML Template Policy

## Purpose
ClearML template tasks are selected by a strict contract. Runtime now looks up templates only by:

- `project`
- `process:<name>`
- `template:true`
- `template_set:<id>`
- `schema:<version>`
- `solution:tabular-analysis`

Fallback lookup without `template_set` or `schema` is intentionally disabled.

Seed pipelines are a separate first-class contract:

- `process:pipeline`
- `task_kind:seed`
- `pipeline_profile:<name>`

These tasks are stored as `TaskTypes.controller` so operators can inspect, clone,
edit, and run them directly from the ClearML Pipelines tab.
They live in the seed pipeline projects `<project_root>/TabularAnalysis/.pipelines/<profile>`.

## Required Config
- `run.clearml.template_usecase_id`
- `run.clearml.template_set_id`
- `run.schema_version`
- `run.clearml.project_root`

`template_set_id` is the primary version key for template generations. `schema_version` is a compatibility tag for artifacts and manifests.

## Commands
Use the canonical entrypoint only:

```bash
python tools/clearml_templates/manage_templates.py --plan --project-root LOCAL
python tools/clearml_templates/manage_templates.py --apply --project-root LOCAL --repo <repo_url> --branch <branch>
python tools/clearml_templates/manage_templates.py --validate --project-root LOCAL --repo <repo_url> --branch <branch>
```

Supported operations are `--plan`, `--apply`, and `--validate`. Historical examples that mention deprecated list/cleanup flows or the old module-style template command are obsolete.

## Operational Rules
- Keep `template_set_id` fixed during normal operation.
- Change `template_set_id` only for intentionally breaking template changes.
- `--apply` is the normalization step for template metadata, script settings, and tags.
- `--apply` also creates or refreshes the seed pipelines for `pipeline`, `train_model_full`, and `train_ensemble_full`.
- `--validate` checks those seed pipeline projects as well.
- When a template generation is intentionally replaced, mark the older template as `template:deprecated` instead of keeping multiple active variants.
- Run `--apply` before `--validate` when tags, script, repo, or branch changed.
- Do not pin runtime to template task IDs; always resolve by canonical tags.

## Source Of Truth
- `conf/clearml/templates.yaml`
- `conf/run/base.yaml`
- `tools/clearml_templates/manage_templates.py`
- `src/tabular_analysis/clearml/templates.py`

