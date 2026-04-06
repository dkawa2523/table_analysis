# 53 ClearML HyperParameters Contract

## 目的

ClearML UI の HyperParameters を、`General` だらけにせず、operator と開発者が読める section に分けて表示することが目的です。

## 背景

現在の pipeline UI 契約では、operator が見る面は 2 つあります。

- `Configuration > OperatorInputs`
  - read-only mirror
- `Hyperparameters`
  - 実行ソースの正本

このドキュメントは後者、つまり task parameter をどう section 分けして残すかの契約です。

## 原則

- full config は `config_resolved.yaml`
- UI には主要入力だけを section ごとに載せる
- template task と run task で意味が変わる値は run 側で上書きする
- seed card と actual run で意味が変わる値は actual run 側で current values へ正規化する
- Hydra に渡す key は plain dotted key を正本にする

## 主な section

- `inputs`
- `dataset`
- `selection`
- `preprocess`
- `model`
- `eval`
- `optimize`
- `pipeline`
- `clearml`

## 例

### pipeline

- `pipeline.profile`
- `pipeline.run_train`
- `pipeline.run_train_ensemble`
- `pipeline.model_set`

### selection

- `pipeline.selection.enabled_preprocess_variants`
- `pipeline.selection.enabled_model_variants`
- `ensemble.selection.enabled_methods`

固定 DAG の seed pipeline では、subset 実行は `selection` section で表現します。  
`pipeline.model_variants` や `pipeline.grid.model_variants` は local / ad hoc 実行の互換用で、operator 向けの通常 UI では graph-shaping key として扱います。

### clearml

- `run.clearml.enabled`
- `run.clearml.execution`
- `run.clearml.project_root`
- `run.clearml.code_ref.*`

## seed card と actual run の違い

| 項目 | seed card | actual run |
| --- | --- | --- |
| `run.usecase_id` | `TabularAnalysis` 既定値を持ってよい | runtime が actual value へ正規化 |
| `data.raw_dataset_id` | placeholder 可 | placeholder 不可 |
| legacy `%2E` key | historical drift として残りうる | current flow では plain dotted key を正本にする |

## operator が通常触る section

- `inputs`
  - `run.usecase_id`
- `dataset`
  - `data.raw_dataset_id`
- `selection`
  - `pipeline.selection.enabled_preprocess_variants`
  - `pipeline.selection.enabled_model_variants`
  - `ensemble.selection.enabled_methods`
- `pipeline`
  - `pipeline.profile`

## current source of truth

- section 分類
  - `conf/clearml/hyperparams_sections.yaml`
- pipeline 側の editable whitelist / `OperatorInputs`
  - `src/tabular_analysis/processes/pipeline_support.py`
- actual task parameter write/reset
  - `src/tabular_analysis/platform_adapter_task_ops.py`

## source of truth

- `src/tabular_analysis/clearml/hparams.py`
- `src/tabular_analysis/platform_adapter_task.py`
- `docs/61_CLEARML_HPARAMS_SECTIONS.md`


