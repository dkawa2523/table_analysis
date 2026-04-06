# 83 Ensemble Policy

## 目的

ensemble を標準運用にどう組み込むかを定義します。

## 方針

- 単体モデルが正本
- ensemble は opt-in
- full ensemble 運用は `LOCAL/TabularAnalysis/.pipelines/train_ensemble_full` の seed pipeline card から `NEW RUN` する
- seed card の `Configuration > OperatorInputs` は確認用 mirror で、実際の `data.raw_dataset_id` や `ensemble.selection.enabled_methods` の編集は `Hyperparameters` 側で行う
- `run.usecase_id` を未編集で起動した場合も、actual run では runtime が一意な usecase に切り替える

## supported methods

- `mean_topk`
- `weighted`
- `stacking`

## 使い分け

### mean_topk

- 最も単純
- baseline として使いやすい

### weighted

- score ベースの重み付け
- 単体より少し強い候補になりやすい

### stacking

- 最も柔軟
- ただし評価設計に注意が必要

## policy

- `ensemble.enabled=false` が既定
- pipeline 既定は単体モデル + leaderboard
- ensemble を回すときだけ `pipeline.run_train_ensemble=true`
- operator の full ensemble 正規入口は `train_ensemble_full` seed card の `NEW RUN`
- subset 実行は `ensemble.selection.enabled_methods` と `pipeline.selection.enabled_model_variants` で表現し、DAG 自体は seed profile 固定のまま維持する


