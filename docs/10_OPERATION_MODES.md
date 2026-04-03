# 10 Operation Modes

`run.clearml.execution` controls how tasks interact with ClearML.

## Modes
- `local`
  - ClearML disabled.
  - Runs everything in the current process tree.
- `logging`
  - Executes locally and logs tasks/artifacts to ClearML.
- `agent`
  - Sends a task to a ClearML agent queue with remote execution.
- `clone`
  - Uses a template task as the remote execution source.
- `pipeline_controller`
  - Clones a visible pipeline template task, runs it as a ClearML controller, and dispatches child tasks through step templates.

## Recommended usage
1. Start with `local` for fast debugging.
2. Use `logging` to verify UI contracts and artifacts.
3. Use `pipeline_controller` for operator-facing pipeline execution from the ClearML Pipelines tab.

## Pipeline behavior
- `local` and `logging` use the local sequential pipeline driver.
- `pipeline_controller` uses a visible controller template as the single source of truth.
- `agent` and `clone` remain supported for direct remote task execution.
