# 17 Operator Quickstart

この 1 ページは、operator が ClearML UI から seed pipeline を開き、`NEW RUN` で安全に実行するための最短手順です。  
詳細は [16_OPERATIONS_RUNBOOK.md](16_OPERATIONS_RUNBOOK.md)、問題切り分けは [69_CLEARML_TROUBLESHOOTING.md](69_CLEARML_TROUBLESHOOTING.md) を参照してください。

## 1. 使う入口

- `LOCAL/TabularAnalysis/.pipelines/pipeline`
- `LOCAL/TabularAnalysis/.pipelines/train_model_full`
- `LOCAL/TabularAnalysis/.pipelines/train_ensemble_full`

使い分け:

| seed profile | 実行内容 |
| --- | --- |
| `pipeline` | preprocess + single-model train + leaderboard |
| `train_model_full` | preprocess + single-model train |
| `train_ensemble_full` | preprocess + single-model train + ensemble + leaderboard |

## 2. UI の見方

| 画面 | 役割 | 編集するか |
| --- | --- | --- |
| `Configuration > OperatorInputs` | grouped mirror | しない |
| `Hyperparameters` | 実編集の正本 | する |

重要:

- current seed / current `NEW RUN` の `Hyperparameters` は plain dotted key が正本です
- `run.usecase_id` や `data.raw_dataset_id` は `Args` の dotted key として見えます
- `%2E` を含む key や、`data.raw_dataset_id` と `data/raw_dataset_id` の重複が見える場合は historical task の可能性があります

## 3. 最短手順

1. ClearML UI で目的の seed card を開く
2. `NEW RUN` を押す
3. `Configuration > OperatorInputs` で既定値を確認する
4. `Hyperparameters` を開く
5. `data.raw_dataset_id` を実在する raw dataset id に置き換える
6. 必要なら `run.usecase_id` や selection 系を更新する
7. 実行する

## 4. 通常編集する項目

- `run.usecase_id`
- `data.raw_dataset_id`
- `pipeline.selection.enabled_preprocess_variants`
- `pipeline.selection.enabled_model_variants`
- `ensemble.selection.enabled_methods`
- `ensemble.top_k`

## 5. 見えていても通常は既定値のまま使う項目

- `pipeline.profile`
- `pipeline.run_*`
- `pipeline.model_set`
- `pipeline.grid.preprocess_variants`
- `pipeline.grid.model_variants`
- `data.split.*`
- `eval.*`

補足:

- これらも `Hyperparameters` に表示されます
- ただし `pipeline.profile` と `pipeline.run_*` は graph-shaping 値なので、通常運用では profile 既定値のまま使います

## 6. 実行時の注意

- seed card の `data.raw_dataset_id=REPLACE_WITH_EXISTING_RAW_DATASET_ID` は正常です
- placeholder のまま実行すると fail-fast します
- `run.usecase_id` を seed 既定値 `TabularAnalysis` のまま起動した場合でも、actual run では runtime が一意値へ自動採番します

## 7. 実行後に見る場所

run controller:

- `LOCAL/TabularAnalysis/Pipelines/Runs/<usecase_id>`

child tasks:

- `LOCAL/TabularAnalysis/Runs/<usecase_id>/01_Datasets`
- `LOCAL/TabularAnalysis/Runs/<usecase_id>/02_Preprocess`
- `LOCAL/TabularAnalysis/Runs/<usecase_id>/03_TrainModels`
- `LOCAL/TabularAnalysis/Runs/<usecase_id>/04_Ensembles`
- `LOCAL/TabularAnalysis/Runs/<usecase_id>/05_Infer`
- `LOCAL/TabularAnalysis/Runs/<usecase_id>/05_Infer_Children`
- `LOCAL/TabularAnalysis/Runs/<usecase_id>/99_Leaderboard`
