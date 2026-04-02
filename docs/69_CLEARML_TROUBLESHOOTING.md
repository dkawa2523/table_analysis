# 69 ClearML Troubleshooting

## Fast recovery path
1. Refresh templates.
2. Validate template lookup.
3. Check the controller task script.
4. Check queue / agent status.

## Template refresh
```bash
python tools/clearml_templates/manage_templates.py --apply --project-root LOCAL
python tools/clearml_templates/manage_templates.py --validate --project-root LOCAL
```

Runtime resolves templates strictly by `project`, `process`, `template_set`,
and `schema`. If lookup fails, fix the template metadata instead of relying on
fallback behavior.

## Common failure modes

### Tasks stay queued
- The queue has no healthy agent.
- The queue name on the task does not match the queue the agent watches.
- A stale failed task is still occupying the expected run slot in the UI.

### Wrong repo / branch / entry point
Inspect the script directly:

```bash
python - <<'PY'
from clearml import Task

task = Task.get_task(task_id="<TASK_ID>")
print(task.get_script())
PY
```

Expected:
- `repository` points to this solution repo
- `entry_point` is `tools/clearml_entrypoint.py`
- `branch` or `version_num` follows `run.clearml.code_ref.*`

### Branch vs commit pin
- Trial and routine operation: `run.clearml.code_ref.mode=branch`
- Pinned production run: `run.clearml.code_ref.mode=commit`

### Child task bootstrap errors
- Missing editable install or missing extras on the agent
- Missing optional libraries for the selected model family
- Agent image missing system packages expected by the entry point bootstrap

## Recommended operator commands
```bash
python tools/rehearsal/run_pipeline_v2.py --execution logging --project-root LOCAL
python tools/tests/rehearsal_verify_clearml_ui.py --usecase-id <USECASE_ID> --project-root LOCAL
```
