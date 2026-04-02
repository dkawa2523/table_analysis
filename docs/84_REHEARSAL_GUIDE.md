# 84 Rehearsal Guide

## Purpose
Use rehearsal to validate the full path from local execution to ClearML-backed
execution with the smallest repeatable workflow.

## Canonical tool
- `tools/rehearsal/run_pipeline_v2.py`
  - Runs `dataset_register -> pipeline`
  - Supports `local`, `logging`, and `agent`
  - Emits `usecase_id`, dataset id, and pipeline task id

## Recommended flow

### 1) Local dry validation
```bash
python tools/rehearsal/run_pipeline_v2.py --execution local \
  --task-type regression --preprocess stdscaler_ohe --models ridge,elasticnet
```

### 2) ClearML logging validation
```bash
python tools/rehearsal/run_pipeline_v2.py --execution logging \
  --task-type regression --preprocess stdscaler_ohe --models ridge,elasticnet \
  --project-root LOCAL
```

### 3) PipelineController + agent validation
```bash
python tools/clearml_templates/manage_templates.py --apply --project-root LOCAL
python tools/clearml_templates/manage_templates.py --validate --project-root LOCAL

python tools/rehearsal/run_pipeline_v2.py --execution agent --queue-name default \
  --task-type regression --preprocess stdscaler_ohe --models ridge,elasticnet \
  --project-root LOCAL
```

## UI verification
```bash
python tools/tests/rehearsal_verify_clearml_ui.py --usecase-id <USECASE_ID> --project-root LOCAL
```

Check:
- project layout
- task grouping
- HyperParameters sections
- `config_resolved.yaml`, `out.json`, `manifest.json`
- leaderboard and pipeline reports

## Important outputs
- `work/rehearsal/out/<mode>/<usecase_id>/rehearsal_summary.json`
- `work/rehearsal/out/<mode>/<usecase_id>/99_pipeline/report.json`
- `work/rehearsal/out/<mode>/<usecase_id>/99_pipeline/report_links.json`

## Source of truth
- `conf/run/base.yaml`
- `conf/clearml/templates.yaml`
- `conf/clearml/project_layout.yaml`
- `docs/67_REHEARSAL_COMMANDS.md`
- `docs/69_CLEARML_TROUBLESHOOTING.md`
