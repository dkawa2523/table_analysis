# 66 Naming Tagging Policy

## 目的

task name、tag、property を predictable にして、UI と自動化の両方から追いやすくするためのポリシーです。

## task name の原則

- process 名が先頭に来る
- variant や preprocess は必要なときだけ suffix として足す

例:

- `dataset_register`
- `preprocess__stdscaler_ohe`
- `train__stdscaler_ohe__lgbm`
- `ensemble__stdscaler_ohe__weighted`

## 必須 tag

- `usecase:<usecase_id>`
- `process:<process>`
- `schema:<schema_version>`

条件付き:

- `grid:<grid_run_id>`
- `retrain:<retrain_run_id>`
- `preprocess:<variant>`
- `model:<variant>`
- `ensemble:<method>`

## template 用 tag

- `template:true`
- `template_set:<id>`
- `task_kind:template`

これらは runtime task に残さないのが原則です。

## seed / run 用 tag

pipeline controller は step template と別契約です。

- seed pipeline
  - `process:pipeline`
  - `task_kind:seed`
  - `pipeline_profile:<name>`
- actual run controller
  - `process:pipeline`
  - `task_kind:run`
  - `usecase:<actual>`

seed pipeline には `template:true` を付けません。step template と seed pipeline を UI 上で混同しないためです。

## property

基本 property:

- `usecase_id`
- `process`
- `schema_version`
- `code_version`
- `platform_version`

追加例:

- `processed_dataset_id`
- `split_hash`
- `model_id`
- `primary_metric`


