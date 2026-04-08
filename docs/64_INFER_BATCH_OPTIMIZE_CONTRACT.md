# 64 Infer Batch Optimize Contract

## 目的

`infer.mode=batch` と `infer.mode=optimize` の契約をまとめます。

## mode 一覧

- `single`
  - 単一入力を推論
- `batch`
  - 複数入力を推論
- `optimize`
  - 入力空間を探索し、目的値に対して最適化

## batch

主な入力:

- `infer.model_id` または `infer.train_task_id`
- `infer.batch.inputs_path`
- `infer.batch.inputs_json`
- `infer.batch.execution`

補足:

- inline JSON batch は `infer.batch.inputs_json` を正本として使う
- `infer.input_json` は single 用
- `infer.batch.execution=clearml_children` は batch executor の 1 実装で、将来別 backend へ差し替え可能な境界として扱う

出力:

- batch prediction artifact
- summary table

## optimize

主な入力:

- `infer.model_id` または `infer.train_task_id`
- `infer.optimize.backend`
- `infer.optimize.n_trials`
- `infer.optimize.direction`
- `infer.optimize.search_space`
- `infer.optimize.objective.key`
- `infer.optimize.sampler`

補足:

- 現在の optimize backend は `optuna`
- `infer.optimize.sampler_name` は廃止し、`infer.optimize.sampler` に統一
- backend 固有実装は infer core から分離し、将来別システムへ置き換え可能な境界として扱う

出力:

- best trial summary
- trial history
- response / importance plot

## ClearML 上の見え方

- summary task は `05_Infer`
- child task があれば `05_Infer_Children`

運用上の推奨:

- cross-agent で安定させたいときは `infer.train_task_id` を優先する
- `model_bundle.joblib` のローカル path は infer 推奨参照として扱わない
- `clearml_children` を使うときは parent と child の queue role を分ける

## bootstrap

- infer は model metadata から必要 extra を解決する
- optimize のときだけ `optuna` を追加する


