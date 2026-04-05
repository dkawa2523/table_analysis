# 63 ClearML Pipelines Visibility

## Goal
Pipeline definitions and runs should always appear in the ClearML UI
as visible controller tasks under their canonical seed and run projects.

## Canonical contract
- `manage_templates.py --apply` creates or updates completed seed pipeline tasks under `LOCAL/.../.pipelines/<profile>`.
- Seed pipelines use `TaskTypes.controller`, `process:pipeline`, `task_kind:seed`, and `pipeline_profile:<name>`.
- Runtime `task=pipeline run.clearml.execution=pipeline_controller` clones a seed pipeline instead of building an ad hoc hidden controller.
- Seed pipelines live under `<project_root>/TabularAnalysis/.pipelines/<profile>`.
- Runtime controller clones live under `<project_root>/TabularAnalysis/Pipelines/Runs/<usecase_id>`.
- Child tasks live under `<project_root>/TabularAnalysis/Runs/<usecase_id>/<process_group>` so the controller run and downstream tasks stay traceable from one root.

## Operator flow
1. Run `python tools/clearml_templates/manage_templates.py --apply --project-root <ROOT>`.
2. Open the seed pipeline project `<ROOT>/TabularAnalysis/.pipelines/<profile>` and confirm the seed pipeline card is visible.
3. Open that seed pipeline card and launch `NEW RUN` when needed.
4. Inspect `Configuration > OperatorInputs` first, verify whether `data.raw_dataset_id` is still the placeholder, then use `Hyperparameters` for the actual edit when needed.
5. Replace `data.raw_dataset_id` with an existing raw dataset id before launch. The cloned run keeps `Hyperparameters` as the source of truth and mirrors the resolved values back into `OperatorInputs`.
6. Run the cloned task, or launch the same profile from CLI with `run.clearml.execution=pipeline_controller`.
7. Verify that the seed card remains visible in `.pipelines/<profile>` and the run controller remains visible in `Pipelines/Runs/<usecase_id>`.

