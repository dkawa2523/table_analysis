# 60 Pipeline Train Contract

## 目的

このドキュメントは、`pipeline` がどのように preprocess、train、ensemble、leaderboard を束ねるかの契約をまとめたものです。

## pipeline の基本入力

- `data.raw_dataset_id`
- `pipeline.selection.enabled_preprocess_variants`
- `pipeline.model_set`
- `pipeline.selection.enabled_model_variants`
- `pipeline.run_train`
- `pipeline.run_train_ensemble`
- `pipeline.run_leaderboard`
- `ensemble.selection.enabled_methods`
- `ensemble.top_k`
- `exec_policy.*`

seed pipeline の正規運用では、dataset 登録は含めず、既存 `raw_dataset_id` を入力にします。
`pipeline.model_variants` と `pipeline.preprocess_variant` は local / ad hoc 実行の互換入力として残りますが、fixed-DAG seed pipeline の operator UI では `selection` 系を使います。
seed card では `Configuration > OperatorInputs` に placeholder `REPLACE_WITH_EXISTING_RAW_DATASET_ID` が見えても正常で、actual run では `Hyperparameters` 側の実値が正本になります。
`run.usecase_id` を seed 既定値 `TabularAnalysis` のまま起動した場合も、actual run では runtime が一意な usecase を採番します。

## 実行フロー

1. raw dataset を取得
2. preprocess 母集合を解決し、selection で有効候補を絞る
3. model 母集合を解決し、selection で有効候補を絞る
4. train jobs を計画
5. 必要なら ensemble jobs を計画
6. leaderboard を実行
7. report / summary を出力

## comparability の条件

train 候補は次が揃っていることを前提に比較します。

- `task_type`
- `primary_metric`
- `direction`
- `split_hash`
- `recipe_hash`

## model set

現在の標準回帰 full set は `regression_all` です。  
canonical 13 モデル:

- `catboost`
- `elasticnet`
- `extra_trees`
- `gaussian_process`
- `gradient_boosting`
- `knn`
- `lasso`
- `lgbm`
- `linear_regression`
- `mlp`
- `random_forest`
- `ridge`
- `xgboost`

`svr` は標準 full set から除外しています。

## ensemble の扱い

- 既定では `pipeline.run_train_ensemble=false`
- operator 向け full ensemble は `train_ensemble_full` seed pipeline が正本
- subset 実行は `ensemble.selection.enabled_methods` で表現
- 3 method:
  - `mean_topk`
  - `weighted`
  - `stacking`

## subset selection

fixed DAG の seed pipeline では、step の追加削除ではなく selection で subset を表現します。

- `pipeline.selection.enabled_preprocess_variants`
- `pipeline.selection.enabled_model_variants`
- `ensemble.selection.enabled_methods`

非選択候補は v1 では child task を作らず、`pipeline_run.json` / `report.json` で `disabled_by_selection` として記録します。

## report

pipeline は最低でも次を出します。

- `pipeline_run.json`
- `report.md`
- `report.json`
- `report_links.json`
- `run_summary.json`

## local / logging / pipeline_controller の違い

- `local`
  - すべてローカル
- `logging`
  - ローカル実行しつつ ClearML に記録
- `pipeline_controller`
  - `.pipelines/<profile>` の seed pipeline card から `NEW RUN` して controller 実行
  - seed card は確認用、実編集は `Hyperparameters`
  - actual run は `Pipelines/Runs/<usecase_id>` と `Runs/<usecase_id>/*` へ切り替わる

## queue ルール

- controller
  - `controller`
- light train / preprocess / leaderboard / ensemble
  - `default`
- heavy train
  - `heavy-model`


