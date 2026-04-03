# 16_OPERATIONS_RUNBOOK (Selection Flow)

## Goal
運用担当者が **pipeline 実行 → leaderboard レビュー → 推論時のモデル選択** まで迷わず進められるよう、
最低限の手順と確認ポイントをまとめます。

## Audience
- 運用担当 / 開発者
- ClearML を使う運用（logging/agent/clone）とローカル運用の両方を対象

## Preconditions
- Python 3.10+ / `uv sync --frozen` (ClearML parity: task-specific `uv sync --extra ... --frozen`)
- データ入力が確定していること（`data.dataset_path` または `data.raw_dataset_id`）
- ClearML を使う場合は `clearml.conf` / 環境変数の設定が完了していること

## Pre-flight Checks (Recommended)
```bash
# 基本チェック（conf/ / platform import / ClearML 接続チェック）
python -m tabular_analysis.doctor

# ClearML agent/clone を使う場合の例
python -m tabular_analysis.doctor \
  run.clearml.enabled=true \
  run.clearml.execution=agent \
  run.clearml.queue_name=default
```

環境が不安定なときは quick verify で最低限の動作確認を行います。
```bash
python tools/tests/verify_all.py --quick
```

## Flow
### 1) dataset_register（raw_dataset_id を取得）
Local mode（ClearML 無効）:
```bash
python -m tabular_analysis.cli task=dataset_register \
  run.clearml.enabled=false \
  run.output_dir=outputs/20260101_120000 \
  data.dataset_path=/path/to/data.csv \
  data.target_column=target
```

ClearML logging mode（ローカル実行 + ログ記録）:
```bash
python -m tabular_analysis.cli task=dataset_register \
  run.clearml.enabled=true \
  run.clearml.execution=logging \
  run.output_dir=outputs/20260101_120000 \
  data.dataset_path=/path/to/data.csv \
  data.target_column=target
```

### 2) Pipeline 実行（train + leaderboard まで）
Local mode（ClearML 無効）:
```bash
python -m tabular_analysis.cli task=pipeline \
  run.clearml.enabled=false \
  run.output_dir=outputs/20260101_120000 \
  data.raw_dataset_id=local:<RAW_DATASET_ID> \
  data.dataset_path=/path/to/data.csv \
  data.target_column=target
```

ClearML logging mode（ローカル実行 + ログ記録）:
```bash
python -m tabular_analysis.cli task=pipeline \
  run.clearml.enabled=true \
  run.clearml.execution=logging \
  run.output_dir=outputs/20260101_120000 \
  data.raw_dataset_id=<RAW_DATASET_ID>
```

Agent 実行は PipelineController を使う。
```bash
python -m tabular_analysis.cli task=pipeline \
  run.clearml.enabled=true \
  run.clearml.execution=pipeline_controller \
  run.clearml.queue_name=default \
  data.raw_dataset_id=<RAW_DATASET_ID>
```

### 2.1) Dry-run（plan だけ確認）
実行前に plan を確認し、タスク数と project 階層を把握します。
```bash
python -m tabular_analysis.cli task=pipeline --dry-run \
  run.clearml.enabled=true \
  run.clearml.execution=pipeline_controller \
  data.raw_dataset_id=<RAW_DATASET_ID>
```

確認ポイント:
- preprocess/train/ensemble の件数
- fail_policy / limits / parallelism の値
- project layout の例が意図通りか

limits を超えた場合は、`pipeline.groups.*` の include/exclude を調整するか、
`pipeline.limits.max_*` を試験時のみ一時的に引き上げます。

### 2.2) Full run（安全に上限を引き上げる）
1) dry-run で plan を確認
2) queue/agent の空きと並列数を確認
3) `pipeline.limits.*` と `pipeline.parallelism.*` を明示指定して実行

例:
```bash
python -m tabular_analysis.cli task=pipeline \
  run.clearml.enabled=true \
  run.clearml.execution=pipeline_controller \
  run.clearml.queue_name=default \
  data.raw_dataset_id=<RAW_DATASET_ID> \
  pipeline.limits.max_preprocess_variants=5 \
  pipeline.limits.max_train_tasks=50 \
  pipeline.limits.max_ensemble_tasks=5 \
  pipeline.parallelism.max_concurrent_steps=6 \
  pipeline.parallelism.max_concurrent_train=4
```

### 2) Leaderboard レビュー
- `outputs/.../05_leaderboard/leaderboard.csv` と `recommendation.json` を確認
- `primary_metric` / `direction` が意図どおりか確認
- `split_hash` / `recipe_hash` が一致しているか確認（比較可能性の保証）

### 3) 推論時のモデル選択
推論時に **ユーザーが model_id を指定**して実行します。

Local mode での例:
```bash
python -m tabular_analysis.cli task=infer \
  run.clearml.enabled=false \
  infer.model_id=outputs/20260101_120000/03_train_model/model_bundle.joblib
```

ClearML registry を使う場合の例:
```bash
python -m tabular_analysis.cli task=infer \
  run.clearml.enabled=true \
  infer.model_id=<CLEARML_REGISTRY_MODEL_ID>
```

## Registry Tag運用（安定運用のための前提）
- `train_model` / `train_ensemble` が **全モデルを registry に登録**する運用
- 絞り込み用タグ:
  - `usecase:<id>` / `dataset:<processed_dataset_id>` / `split:<split_hash>` / `recipe:<recipe_hash>`
  - `preprocess:<variant>` / `model_variant:<variant>` / `task_type:<type>`
  - `task:pipeline:<id>` / `task:preprocess:<id>` / `task:train_model:<id>` / `task:train_ensemble:<id>`
- 推薦モデルの絞り込み:
  - `leaderboard:recommended` + `recommend_rank:<n>`
  - スコアは metadata に保存（tagでは保持しない）
- Tag上限対策（ensemble）:
  - `train_ensemble.registry.tag_limits.train_task_ids` で tag化する train_task_id の上限を設定
  - 全ID一覧が必要な場合は `train_ensemble.registry.metadata.full_train_task_ids=true` を使用

## Troubleshooting
- `model_bundle.joblib` が見つからない: train 出力か ClearML の artifact を確認
- registry 登録失敗: ClearML 認証 / ネットワーク / registry 側の権限を確認
- UI 契約が疑わしい: `python -m tabular_analysis.doctor --lint-run <output_dir>` を実行

## Outputs
- 共通: `config_resolved.yaml`, `out.json`, `manifest.json`
- pipeline: `pipeline_run.json`, `report.md`
