# 50 ClearML Processed Dataset Contract

## 目的

preprocess の結果を ClearML Dataset として登録し、train / leaderboard / infer が同じ processed dataset を参照できるようにする契約です。

## preprocess が出すもの

- `processed_dataset_id`
- `split_hash`
- `recipe_hash`
- preprocess bundle
- split / schema / recipe artifact

## dataset 名の考え方

processed dataset は usecase と preprocess variant を含む名前で管理します。  
schema version は suffix / tag で明示します。

## dataset に含める artifact

- `split.json`
- `schema.json`
- `recipe.json`
- `preprocess_bundle.*`
- `meta.json`

必要に応じて:

- `X.parquet`
- `y.parquet`
- `feature_names.json`

## train / infer での扱い

- train は `processed_dataset_id` を読み、local copy を取得する
- infer は model bundle から preprocess 情報を読むのが主経路
- ただし comparability の確認では processed dataset の identity が重要

## comparability

次のいずれかで比較可能性を判断します。

- 同じ `processed_dataset_id`
- 同じ `split_hash + recipe_hash`

## failure policy

ClearML Dataset 登録に失敗した場合:

- `run.clearml.enabled=false`
  - local only で継続可
- `run.clearml.enabled=true`
  - 契約違反として fail を推奨


