# 53 ClearML HyperParameters Contract

## 目的

ClearML UI の HyperParameters を、`General` だらけにせず、operator と開発者が読める section に分けて表示することが目的です。

## 原則

- full config は `config_resolved.yaml`
- UI には主要入力だけを section ごとに載せる
- template task と run task で意味が変わる値は run 側で上書きする

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

## source of truth

- `src/tabular_analysis/clearml/hparams.py`
- `src/tabular_analysis/platform_adapter_task.py`
- `docs/61_CLEARML_HPARAMS_SECTIONS.md`


