# 82 ClearML Project Layout

## Purpose
ClearML project naming is config-driven so every task lands in a predictable
location in the UI.

## Canonical shape
`<ROOT>/<solution_root>/Runs/<usecase_id>/<process_group>`

Visible pipeline controllers use dedicated roots:

- pipeline seed: `<ROOT>/<solution_root>/.pipelines/<profile>`
- pipeline run: `<ROOT>/<solution_root>/Pipelines/Runs/<usecase_id>`
- step template: `<ROOT>/<solution_root>/Templates/Steps/<process_group>`

- `ROOT`: `run.clearml.project_root`
- `solution_root`: `run.clearml.project_layout.solution_root`
- `usecase_id`: `run.usecase_id`
- `process_group`: `run.clearml.project_layout.group_map[process]`

Notes:
- seed card は `.pipelines/<profile>` に固定で置かれるため、seed 既定値 `run.usecase_id=TabularAnalysis` を持っていても project path は変わりません
- actual run では `run.usecase_id` がそのまま使われるか、seed 既定値のままなら runtime が一意な `<usecase_id>` を採番します
- child task は常に `Runs/<usecase_id>/<process_group>` 配下へ着地します

## Typical groups
- `01_Datasets`
- `02_Preprocess`
- `03_TrainModels`
- `04_Ensembles`
- `05_Infer`
- `05_Infer_Children`
- `99_Leaderboard`
- `.pipelines/<profile>`
- `Pipelines/Runs/<usecase_id>`
- `Templates/Steps/<group>`

## Source of truth
- `conf/run/base.yaml`
- `conf/clearml/project_layout.yaml`
- `conf/clearml/templates.yaml`

## Review guidance
- Compare actual UI placement against `docs/03_CLEARML_UI_CONTRACT.md`.
- Use `docs/43_CLEARML_UI_LAYOUT_EXAMPLES.md` as a lightweight visual reference.
- Record differences in the PR description, an operations note, or your team's issue tracker.

