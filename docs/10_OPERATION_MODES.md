# 10 Operation Modes

`run.clearml.execution` は、同じ task をどの運用形態で動かすかを決めるスイッチです。

## 一覧

| mode | 主用途 | 実行場所 | ClearML 上の見え方 |
| --- | --- | --- | --- |
| `local` | 最速の開発確認 | ローカル | 記録しない |
| `logging` | UI 契約確認 | ローカル | task / artifact / plots を記録 |
| `agent` | 単体 task の remote 実行 | Agent | queue ベースの task |
| `clone` | 既存 task を元に remote 実行 | Agent | task clone |
| `pipeline_controller` | operator 向け pipeline 実行 | Agent | `.pipelines/<profile>` の seed pipeline card から `NEW RUN` |

## 推奨の使い分け

### 開発者

1. `local`
2. `logging`
3. 必要時だけ `pipeline_controller`

### operator

1. `manage_templates --apply`
2. `manage_templates --validate`
3. ClearML UI で `.pipelines/<profile>` の seed pipeline card を開き、`NEW RUN` 後に `Configuration > OperatorInputs` を確認し、実値は `Hyperparameters` で更新して実行

## 各モードの詳細

### local

- ClearML なし
- 一番速い
- task 本体のロジック確認に向く

例:

```bash
python -m tabular_analysis.cli task=pipeline \
  run.clearml.enabled=false \
  data.dataset_path=/path/to/data.csv \
  data.target_column=target
```

### logging

- 実行はローカル
- ClearML に metadata / artifacts / plots を残す
- UI 契約確認に向く

例:

```bash
python -m tabular_analysis.cli task=pipeline \
  run.clearml.enabled=true \
  run.clearml.execution=logging \
  run.clearml.project_root=LOCAL \
  data.raw_dataset_id=<RAW_DATASET_ID>
```

### agent

- 単一 task を queue に送る
- task ごとの remote 実行確認に向く

例:

```bash
python -m tabular_analysis.cli task=train_model \
  run.clearml.enabled=true \
  run.clearml.execution=agent \
  run.clearml.queue_name=default \
  train.inputs.preprocess_run_dir=/path/to/preprocess_run \
  group/model=ridge
```

### clone

- template task を元に remote 実行する
- child template の再利用確認に向く

### pipeline_controller

- `.pipelines/<profile>` の seed pipeline card から `NEW RUN` して controller として実行する
- `Configuration > OperatorInputs` では placeholder や最小入力を確認する
- 実際の `run.usecase_id` / `data.raw_dataset_id` は `Hyperparameters` 側で更新する
- `data.raw_dataset_id` は placeholder のままにせず、既存 raw dataset id へ置き換える
- operator 向けの正本
- `controller/default/heavy-model` の queue 分割前提

例:

```bash
python -m tabular_analysis.cli task=pipeline \
  run.clearml.enabled=true \
  run.clearml.execution=pipeline_controller \
  run.clearml.project_root=LOCAL \
  data.raw_dataset_id=<RAW_DATASET_ID> \
  pipeline.selection.enabled_preprocess_variants=[stdscaler_ohe] \
  pipeline.selection.enabled_model_variants=[ridge,lgbm,xgboost]
```

- `run.clearml.queue_name` は pipeline child routing の正本ではありません。
- pipeline mode では `exec_policy.queues.*` が child task の queue を決めます。
- `run.clearml.queue_name` を渡す場合は controller queue の明示用途に限定します。
- seed pipeline の標準運用は `pipeline.run_dataset_register=false` 前提で、入力は既存 `data.raw_dataset_id` です。

## queue との関係

標準の queue 分割:

- `controller`
  - controller
- `default`
  - preprocess、light train、leaderboard、ensemble、infer
- `heavy-model`
  - `catboost`、`xgboost`

## どのモードを正本にするか

- 開発の正本: `local`
- UI 契約の正本: `logging`
- 運用の正本: `pipeline_controller`

## 関連ドキュメント

- [16_OPERATIONS_RUNBOOK.md](16_OPERATIONS_RUNBOOK.md)
- [67_REHEARSAL_COMMANDS.md](67_REHEARSAL_COMMANDS.md)
- [81_CLEARML_TEMPLATE_POLICY.md](81_CLEARML_TEMPLATE_POLICY.md)
- [82_CLEARML_PROJECT_LAYOUT.md](82_CLEARML_PROJECT_LAYOUT.md)

