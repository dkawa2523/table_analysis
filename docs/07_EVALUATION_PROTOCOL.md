# 07_EVALUATION_PROTOCOL（比較可能性の規約）

## 比較可能性（train vs train）
leaderboard は、次の条件が一致する train だけを同じ土俵で比較します。

一致必須：
- `processed_dataset_id`
- `split_hash`
- `eval.task_type`
- `eval.primary_metric` と `eval.direction`
- `eval.seed`（必要なら）

## split の責務
- split は **preprocess の責務**
- train_model は split を再生成しない（再現性と比較可能性のため）

## 不一致の扱い
- `leaderboard.require_comparable=true` の場合：不一致の train を除外し、`excluded_count` を出力
- false の場合：比較は行うが `warning` を out.json/summary.md に必ず残す
