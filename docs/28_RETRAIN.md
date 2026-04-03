# 28 Retrain

## 目的

`retrain` task は、新しいデータに対して pipeline を再実行し、次に使うべき challenger を decision artifact として残すための wrapper です。

## やること

1. 最新データで pipeline を回す
2. leaderboard recommendation を読む
3. `retrain_decision.json` を残す

## 入力

- `data.dataset_path`
- `data.raw_dataset_id`

## 出力

- `retrain_summary.md`
- `retrain_decision.json`
- `retrain_run.json`

参照先として:

- `99_pipeline/pipeline_run.json`
- `05_leaderboard/recommendation.json`

## ClearML traceability

- `retrain:<retrain_run_id>` tag
- `retrain_run_id` property

## 使いどころ

- 定期再学習
- drift / quality の後続アクション
- operator が challenger を比較したいとき

## 例

```bash
python -m tabular_analysis.cli task=retrain \
  run.clearml.enabled=false \
  data.dataset_path=/path/to/latest.csv \
  data.target_column=target
```

