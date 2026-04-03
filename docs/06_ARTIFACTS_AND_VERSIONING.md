# 06 Artifacts And Versioning

## 目的

この repo では、各 task の結果を単なる一時ファイルではなく「再現可能な成果物」として扱います。  
その中心が artifact、`out.json`、`manifest.json`、hash 群です。

## 全 task 共通の基本出力

### `config_resolved.yaml`

- 実行時に解決された Hydra config
- 完全な設定の正本

### `out.json`

- task の結果を他 task が読むための最小契約
- 下流 task は基本的にここを見る

### `manifest.json`

- 入出力、hash、version、親 task、runtime 情報の総覧
- audit と再現確認に使う

## artifact の考え方

artifact は「その task を再利用・検証するために必要な実体」です。

例:

- preprocess
  - `preprocess_bundle.*`
  - `recipe.json`
  - `split.json`
- train_model
  - `model_bundle.*`
  - `metrics.json`
  - `preds_valid.parquet`
- leaderboard
  - `leaderboard.csv`
  - `recommendation.json`
- pipeline
  - `pipeline_run.json`
  - `report.json`
  - `report_links.json`

## hash の役割

### `config_hash`

- 実行設定の識別子

### `split_hash`

- split 設定と split 内容の識別子
- 比較可能性の中心

### `recipe_hash`

- 前処理 recipe の識別子
- preprocess bundle の一意性に使う

### `schema_hash`

- schema の識別子

## versioning の考え方

### `run.schema_version`

- artifact / manifest 契約の互換性バージョン
- tag は `schema:v1` の形で ClearML にも反映される

### code / platform version

- 実行時に code version、platform version を解決して記録する
- manifest と ClearML properties に残す

## どこに書かれるか

### ローカル

- `run.output_dir/<task stage>/...`

### ClearML

- artifact upload
- user properties
- task metadata

## task 間の読み取り方

推奨される読み順:

1. `out.json`
2. 必要なら task artifact
3. 監査や再現時に `manifest.json`

## 比較可能性と artifact

特に leaderboard では、候補同士が次を満たす必要があります。

- 同じ `processed_dataset_id` か、少なくとも同じ `split_hash` と `recipe_hash`
- 同じ `task_type`
- 同じ `primary_metric` と `direction`

## 関連コード

- `src/tabular_analysis/processes/lifecycle.py`
- `src/tabular_analysis/platform_adapter_artifacts.py`
- `src/tabular_analysis/platform_adapter_task_ops.py`


