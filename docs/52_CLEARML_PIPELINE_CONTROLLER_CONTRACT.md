# 52 ClearML Pipeline Controller Contract

## 目的

`task=pipeline` を ClearML 上で operator が理解しやすい controller task として扱うための契約です。

## 正本の考え方

pipeline 実行の正本は visible pipeline template です。  
runtime は hidden controller を ad hoc に作らず、visible template を clone して run controller を作ります。

## template 側

visible pipeline template は次を満たします。

- `TaskTypes.controller`
- `process:pipeline`
- `task_kind:template`
- `pipeline_profile:<name>`
- visible project: `<project_root>/TabularAnalysis/Pipelines`

## run controller 側

run controller は template clone 後に runtime metadata へ上書きされます。

必須の考え方:

- `task_kind:run`
- `usecase:<actual>`
- `template:true` は残さない
- project は template と同じ visible pipeline project

## child task 側

child task は runtime identity を再構築します。

例:

- `process:preprocess`
- `preprocess:stdscaler_ohe`
- `process:train_model`
- `model:lgbm`
- `grid:<grid_run_id>`

template 用 tag や stale usecase は残しません。

## queue 契約

- controller
  - `services`
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


