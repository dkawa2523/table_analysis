# ClearML: Processed Dataset 管理（契約） v2

## 目的
- **前処理結果（processed data）を ClearML Dataset として登録**し、学習・推論で **同一の processed dataset を復元可能**にする。
- split（train/val/test）を再生成せず、**split を Dataset に同梱して固定**する。
- 前処理パイプライン（bundle）も Dataset に同梱し、推論時に必ず同一の変換で処理できるようにする。
- **設計の冗長化を避ける**: ClearML Dataset 取扱いは `tabular_analysis/clearml/datasets.py` に集約し、各processは薄い呼び出しにする。

## 1. Preprocess 出力（必須）
Preprocess Task は以下を満たす。

### 1) ClearML Dataset（processed）
- dataset name: `processed__{usecase_id}__{preprocess_variant}__{split_hash}__{schema_version}`
- tags（例）: `usecase:{usecase_id}`, `process:preprocess`, `type:processed`, `schema:{schema_version}`
- parent: raw dataset を parent として参照（可能なら）

> 注意: 大規模データへの拡張を阻害しないため、`ops.processed_dataset.store_features` を導入し、
> - true: X/y を含める（試験段階のデフォルト）
> - false: bundle+recipe+splits だけを登録（巨大データ対策）
> を選べるようにする。

### 2) Dataset に含めるファイル（必須）
- `split.json`：split index/fold情報
- `schema.json`：入力スキーマ（列・型・カテゴリ）
- `recipe.json`：前処理レシピ（人が読める + 検証用）
- `preprocess_bundle.joblib`：transformer/encoder 等（fit済み）
- `meta.json`：hash群・作成日時・rows/features・store_features

### 3) Dataset に含めるファイル（store_features=true の場合）
- `X.parquet`（または `.npy`）：前処理後特徴量
- `y.parquet`：目的変数
- `feature_names.json`（任意だが推奨）

### 4) Task out.json / manifest.json（必須キー）
out.json:
- `processed_dataset_id`
- `processed_dataset_version`（取得できる場合）
- `split_hash`, `recipe_hash`, `schema_hash`
- `n_rows`, `n_features`
manifest.json:
- inputs: `raw_dataset_id`, `upstream_task_ids`, `store_features`
- outputs: `processed_dataset_id`, `processed_dataset_version`
- hashes: `split_hash`, `recipe_hash`, `schema_hash`, `config_hash`

## 2. Train / Infer の入力契約
- Train/Infer は `processed_dataset_id` を入力として受け取り、**Dataset.get_local_copy() で復元**する。
- Train は split を Dataset の `split.json` から復元し、再分割しない。
- Infer は model_bundle 内の `preprocess_bundle` を優先し、processed dataset の bundle と一致するか検証（warn or fail）。
  - 一致判定は `recipe_hash` / `bundle_hash` を推奨。

## 3. 比較可能性（leaderboard）
- 2つの train は、少なくとも以下が一致したときのみ比較可能:
  - `processed_dataset_id`（または `split_hash` + `recipe_hash` の両方）
  - `primary_metric` と `direction`
  - `task_type`（classification/regression）

## 4. 失敗時の扱い
- ClearML Dataset 登録に失敗した場合:
  - `run.clearml.enabled=false` の場合はローカル保存へフォールバックして OK
  - enabled=true の場合は **明確に fail**し、doctor/contract lint で検出可能にする
