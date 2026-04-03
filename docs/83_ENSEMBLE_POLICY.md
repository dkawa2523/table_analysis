# 83 Ensemble Policy

## 目的

ensemble を標準運用にどう組み込むかを定義します。

## 方針

- 単体モデルが正本
- ensemble は opt-in
- full ensemble 運用は visible template `train_ensemble_full` を使う

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


