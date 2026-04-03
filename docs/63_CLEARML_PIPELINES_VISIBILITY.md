# 63 ClearML Pipelines Visibility

## Goal
Pipeline definitions and runs should always appear in the ClearML Pipelines tab
as visible controller tasks under the target project.

## Canonical contract
- `manage_templates.py --apply` creates or updates visible pipeline template tasks.
- Pipeline templates use `TaskTypes.controller`, `process:pipeline`, `task_kind:template`, and `pipeline_profile:<name>`.
- Runtime `task=pipeline run.clearml.execution=pipeline_controller` clones a visible pipeline template task instead of building an ad hoc hidden controller.
- Template tasks and run controller tasks are created in the same visible pipeline project.
- The canonical pipeline project is `<project_root>/TabularAnalysis/Pipelines`.
- Child tasks live under `<project_root>/TabularAnalysis/Pipelines/<usecase_id>/<process_group>` so the controller and its children stay traceable from one root.

## Operator flow
1. Run `python tools/clearml_templates/manage_templates.py --apply --project-root <ROOT>`.
2. Open the ClearML Pipelines tab and confirm the pipeline template task is visible.
3. Clone or edit that pipeline template task from the UI when needed.
4. Run the cloned task, or launch the same profile from CLI with `run.clearml.execution=pipeline_controller`.
5. Verify that both the template and the run controller remain visible in the project and the Pipelines tab.

