# 61 ClearML HyperParameters Sections

## 目的

pipeline seed と `NEW RUN` の `Hyperparameters` を、第三者がそのまま読める plain dotted key に揃えるための契約です。

## 現在の正本

- `Hyperparameters`
  - 実編集の正本
  - current seed / current `NEW RUN` は plain dotted key を使う
- `Configuration > OperatorInputs`
  - grouped mirror
  - 確認用であり、source of truth ではない

## 基本ルール

- `Args` が current pipeline task の source of truth
- `%2E` や slash 混在は current 正常系では出さない
- named Hyperparameters section は pipeline seed / visible run clone では冗長なので正本にしない
- grouped な見え方が必要な情報は `OperatorInputs` にだけ残す

## 代表キー

主要入力:

- `run.usecase_id`
- `data.raw_dataset_id`
- `pipeline.selection.enabled_preprocess_variants`
- `pipeline.selection.enabled_model_variants`
- `ensemble.selection.enabled_methods`
- `ensemble.top_k`

current seed で見える代表値:

- `data.target_column`
- `data.split.strategy`
- `data.split.test_size`
- `data.split.seed`
- `eval.primary_metric`
- `eval.direction`
- `eval.task_type`
- `eval.cv_folds`
- `pipeline.profile`
- `pipeline.model_set`
- `pipeline.grid.preprocess_variants`
- `pipeline.grid.model_variants`

bootstrap:

- `task`
- `run.clearml.enabled`
- `run.clearml.execution`
- `run.clearml.project_root`
- `run.schema_version`
- `run.clearml.env.*`

## どう読むか

`Hyperparameters`:

- `run.usecase_id`
- `data.raw_dataset_id`
- `pipeline.selection.enabled_model_variants`
- `ensemble.top_k`

`OperatorInputs`:

- `run { usecase_id = ... }`
- `data { raw_dataset_id = ... }`
- `pipeline { profile, selection, grid, model_set }`
- `ensemble { selection, top_k }`
- `eval { ... }`

## 運用上の扱い

通常編集する:

- `run.usecase_id`
- `data.raw_dataset_id`
- `pipeline.selection.enabled_preprocess_variants`
- `pipeline.selection.enabled_model_variants`
- `ensemble.selection.enabled_methods`
- `ensemble.top_k`

通常は既定値のまま使う:

- `pipeline.profile`
- `pipeline.model_set`
- `pipeline.grid.*`
- `data.split.*`
- `eval.*`

fixed DAG internal values:

- `pipeline.run_*`
- `pipeline.plan_only`

これらは operator 用 `Hyperparameters` には出さず、seed profile と runtime 正規化で決まります。

## Source Of Truth

- operator-facing UI contract
  - `src/tabular_analysis/clearml/pipeline_ui_contract.py`
- pipeline UI payload / validation
  - `src/tabular_analysis/processes/pipeline_support.py`
- current task UI sync
  - `src/tabular_analysis/processes/pipeline.py`
- seed publish / drift validate / cleanup
  - `tools/clearml_templates/seed_publish.py`
  - `tools/clearml_templates/drift_validate.py`
  - `tools/clearml_templates/stale_cleanup.py`
