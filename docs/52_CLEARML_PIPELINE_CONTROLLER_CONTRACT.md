# 52 ClearML Pipeline Controller Contract

## 目的

`task=pipeline` を ClearML 上で operator が理解しやすい controller task として扱うための契約です。

## 正本の考え方

pipeline 実行の正本は seed pipeline です。  
runtime は hidden controller を ad hoc に作らず、seed pipeline から run controller を起動します。

## seed pipeline 側

seed pipeline は次を満たします。

- `TaskTypes.controller`
- `process:pipeline`
- `task_kind:seed`
- `pipeline_profile:<name>`
- project: `<project_root>/TabularAnalysis/.pipelines/<profile>`

## run controller 側

run controller は seed pipeline から起動された後に runtime metadata へ上書きされます。

必須の考え方:

- `task_kind:run`
- `usecase:<actual>`
- `template:true` は残さない
- project は `<project_root>/TabularAnalysis/Pipelines/Runs/<usecase_id>`

## child task 側

child task は runtime identity を再構築します。

例:

- `process:preprocess`
- `preprocess:stdscaler_ohe`
- `process:train_model`
- `model:lgbm`
- `grid:<grid_run_id>`

step template tag や stale usecase は残しません。

## queue 契約

- controller
  - `controller`
- light child
  - `default`
- heavy child
  - `heavy-model`

## artifact

controller は次を出します。

- `pipeline_run.json`
- `report.md`
- `report.json`
- `report_links.json`
- `run_summary.json`


