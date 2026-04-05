# 05 Process Catalog

各 task は `python -m tabular_analysis.cli task=<task>` で実行します。  
全 task 共通で `config_resolved.yaml`、`out.json`、`manifest.json` を出力します。

## dataset_register

### 役割

- 入力データを raw dataset として確定する
- schema、preview、基本品質情報を揃える

### 主な入力

- `data.dataset_path`
- `data.target_column`
- `run.clearml.*`

### 主な出力

- `out.json`
  - `raw_dataset_id`
  - `raw_schema`
- artifact
  - `schema.json`
  - `preview.csv`

### どんなときに使うか

- pipeline 実行前に raw dataset を登録したいとき
- dataset path から ClearML Dataset ID を得たいとき

### 主なコード

- `src/tabular_analysis/processes/dataset_register.py`

## preprocess

### 役割

- split を作る
- feature type を推定する
- 変換 recipe と preprocess bundle を作る

### 主な入力

- `data.raw_dataset_id` または `data.dataset_path`
- `preprocess.variant`
- `data.split.*`

### 主な出力

- `out.json`
  - `processed_dataset_id`
  - `preprocess_variant`
  - `split_hash`
  - `recipe_hash`
- artifact
  - `preprocess_bundle.*`
  - `recipe.json`
  - `summary.md`
  - `schema.json`
  - `split.json`

### どんなときに使うか

- 単体で前処理レシピを検証したいとき
- 同じ preprocess を複数モデルで共有したいとき

### 主なコード

- `src/tabular_analysis/processes/preprocess.py`
- `src/tabular_analysis/feature_engineering/`
- `src/tabular_analysis/common/feature_types.py`

## train_model

### 役割

- 単体モデルを学習する
- validation prediction、metrics、bundle を出す

### 主な入力

- `train.inputs.preprocess_run_dir`
- `data.processed_dataset_id`
- `group/model=<variant>`
- `eval.*`

### 主な出力

- `out.json`
  - `train_task_id`
  - `model_id`
  - `best_score`
  - `primary_metric`
  - `task_type`
  - `n_classes`
- artifact
  - `metrics.json`
  - `preds_valid.parquet`
  - `model_bundle.*`
  - `classes.json` for classification

### どんなときに使うか

- 単一モデルを明示的に比較したいとき
- optional model の動作確認をしたいとき

### 主なコード

- `src/tabular_analysis/processes/train_model.py`
- `src/tabular_analysis/registry/models.py`

## train_ensemble

### 役割

- 既存 train 結果を集めて ensemble を構築する

### 主な入力

- `run.usecase_id`
- `preprocess.variant`
- `ensemble.methods`
- `ensemble.top_k`
- `ensemble.selection_metric`
- `ensemble.exclude_variants`

### 主な前提

- 候補 run が `preds_valid.parquet` を持っていること
- local rerun fallback は削除済み

### 主な出力

- `out.json`
  - `train_task_id`
  - `model_id`
  - `best_score`
  - `primary_metric`
- artifact
  - `metrics.json`
  - `ensemble_spec.json`
  - `model_bundle.joblib`

### どんなときに使うか

- 単体モデルだけではなく ensemble まで比較対象に含めたいとき

### 主なコード

- `src/tabular_analysis/processes/train_ensemble.py`

## leaderboard

### 役割

- 候補 train task を比較し、推奨モデルを決める

### 主な入力

- `leaderboard.train_task_ids`
- `leaderboard.train_run_dirs`

### 主な出力

- `out.json`
  - `leaderboard_csv`
  - `recommended_train_task_id`
  - `recommended_model_id`
  - `excluded_count`
- artifact
  - `leaderboard.csv`
  - `recommendation.json`
  - `summary.md`

### どんなときに使うか

- 全候補の比較結果を 1 か所に集約したいとき
- pipeline の最終推奨を確認したいとき

### 主なコード

- `src/tabular_analysis/processes/leaderboard.py`

## infer

### 役割

- 学習済みモデルで推論する

### 主な入力

- `infer.model_id` または `infer.train_task_id`
- `infer.mode=single|batch|optimize`

### 主な出力

- `out.json`
  - `predictions_path`
  - `prediction_path` in single mode

### どんなときに使うか

- registry model から推論したいとき
- bundle path を使ってローカル推論したいとき

### 主なコード

- `src/tabular_analysis/processes/infer.py`
- `src/tabular_analysis/processes/infer_support.py`

## pipeline

### 役割

- 上記 task をまとめて計画・実行する

### 主な入力

- `data.raw_dataset_id`
- `pipeline.selection.enabled_preprocess_variants`
- `pipeline.model_set`
- `pipeline.selection.enabled_model_variants`
- `pipeline.run_*`
- `ensemble.selection.enabled_methods`
- `ensemble.top_k`
- `exec_policy.*`

### 主な出力

- `pipeline_run.json`
- `report.md`
- `report.json`
- `report_links.json`
- `run_summary.json`

### どんなときに使うか

- 前処理から leaderboard までを一括で回したいとき
- seed pipeline から controller 実行したいとき

### 主なコード

- `src/tabular_analysis/processes/pipeline.py`
- `src/tabular_analysis/clearml/pipeline_templates.py`

## retrain

### 役割

- pipeline 結果を踏まえて再学習判断を行う薄い wrapper

### 主なコード

- `src/tabular_analysis/processes/retrain.py`

## task をどう選ぶか

### 単体動作を確かめたい

- `dataset_register`
- `preprocess`
- `train_model`
- `infer`

### 一括で比較評価したい

- `pipeline`

### ensemble だけ追加で試したい

- `train_ensemble`

### 既存候補から推奨を出したい

- `leaderboard`

