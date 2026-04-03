# 60 Pipeline Train Contract

## 目的

このドキュメントは、`pipeline` がどのように preprocess、train、ensemble、leaderboard を束ねるかの契約をまとめたものです。

## pipeline の基本入力

- `data.raw_dataset_id`
- `pipeline.preprocess_variant` / `pipeline.preprocess_variants`
- `pipeline.model_set` / `pipeline.model_variants`
- `pipeline.run_train`
- `pipeline.run_train_ensemble`
- `pipeline.run_leaderboard`
- `ensemble.*`
- `exec_policy.*`

## 実行フロー

1. raw dataset を取得
2. preprocess variant を展開
3. model set / model variants を展開
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
- operator 向け full ensemble は `train_ensemble_full` template が正本
- 3 method:
  - `mean_topk`
  - `weighted`
  - `stacking`

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
  - visible pipeline template を clone して controller 実行

## queue ルール

- controller
  - `services`
- light train / preprocess / leaderboard / ensemble
  - `default`
- heavy train
  - `heavy-model`


