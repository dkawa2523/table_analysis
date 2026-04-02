# 63 ClearML Pipelines Visibility

## Goal
Pipeline runs should appear in the ClearML Pipelines tab with a visible project
and a controller task that matches the runtime contract.

## Current contract
- Use `run.clearml.execution=pipeline_controller`
- The controller task uses `TaskTypes.controller` and receives the `pipeline` system tag
- Pipeline projects are normalized through `run.clearml.pipeline.*`
- Templates must be normalized with `tools/clearml_templates/manage_templates.py --apply`

## Operator flow
1. Refresh templates with `--apply`
2. Start at least one agent on the target queue
3. Run the pipeline with `run.clearml.execution=pipeline_controller`
4. Verify the controller project appears in the Pipelines tab
