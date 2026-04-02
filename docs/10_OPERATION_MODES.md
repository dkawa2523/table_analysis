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
  - Runs the pipeline controller in ClearML and dispatches child tasks through templates.

## Recommended usage
1. Start with `local` for fast debugging.
2. Use `logging` to verify UI contracts and artifacts.
3. Use `pipeline_controller` when validating real agent orchestration.

## Pipeline behavior
- `local` and `logging` use the local sequential pipeline driver.
- `pipeline_controller` uses the controller path.
- `agent` and `clone` remain supported for direct remote task execution.
