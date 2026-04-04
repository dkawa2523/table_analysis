# 22 Execution Policy

## 目的

`exec_policy` は pipeline 実行時の queue、limit、heavy model routing、selection を制御する設定です。  
個々の task 実装に散らさず、実行ポリシーを 1 か所で読めるようにしています。

## source of truth

- `conf/exec_policy/base.yaml`
- `src/tabular_analysis/processes/pipeline.py`

## 主な設定

### limits

- `exec_policy.limits.max_jobs`
  - pipeline が計画できる train job の上限
- `exec_policy.limits.max_models`
  - report / leaderboard に出す上限
- `exec_policy.limits.max_hpo_trials`
  - model ごとの HPO 試行回数

## queues

- `exec_policy.queues.default`
  - 基本 queue
- `exec_policy.queues.pipeline`
  - controller queue
- `exec_policy.queues.train_model_heavy`
  - heavy model queue
- `exec_policy.queues.model_variants`
  - variant 個別 override
- `exec_policy.queues.heavy_model_variants`
  - heavy queue に送る variant 一覧

現在の標準:

- `controller`
  - pipeline controller
- `default`
  - preprocess、light train、leaderboard、ensemble、infer
- `heavy-model`
  - `catboost`、`xgboost`

## selection

- `exec_policy.selection.calibration`
- `exec_policy.selection.uncertainty`
- `exec_policy.selection.ci`

重い機能を標準では無効にし、必要時だけ opt-in するためのフラグです。

## pipeline との関係

`exec_policy` は「何をどれだけ実行してよいか」を決めます。  
実際の candidate 展開は pipeline の plan builder が行います。

### 典型例

- model set を展開する
- heavy model を `heavy-model` に送る
- `max_jobs` を超える場合は plan 側で制限する

## よく使う override 例

```bash
python -m tabular_analysis.cli task=pipeline \
  run.clearml.enabled=true \
  run.clearml.execution=pipeline_controller \
  exec_policy.queues.default=default \
  exec_policy.queues.train_model_heavy=heavy-model \
  exec_policy.queues.heavy_model_variants=[catboost,xgboost] \
  exec_policy.limits.max_jobs=20
```

## heavy model routing

現在の標準 heavy model:

- `catboost`
- `xgboost`

`lgbm` は heavy queue ではなく `default` に残します。

## 運用上の考え方

- queue 分割は operator 視点で理解しやすい単位にする
- HPO や uncertainty のような重い処理は selection で明示制御する
- limit 超過は hidden failure にせず report / run_summary に残す


