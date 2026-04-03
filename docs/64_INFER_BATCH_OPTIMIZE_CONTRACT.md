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

- `infer.batch.inputs_path`
- `infer.batch.inputs_json`

出力:

- batch prediction artifact
- summary table

## optimize

主な入力:

- `infer.optimize.n_trials`
- `infer.optimize.direction`
- `infer.optimize.search_space`
- `infer.optimize.objective.key`

出力:

- best trial summary
- trial history
- response / importance plot

## ClearML 上の見え方

- summary task は `05_Infer`
- child task があれば `05_Infer_Children`

## bootstrap

- infer は model metadata から必要 extra を解決する
- optimize のときだけ `optuna` を追加する


