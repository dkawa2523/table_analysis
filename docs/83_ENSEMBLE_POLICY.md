# 83_ENSEMBLE_POLICY（mean_topk / weighted / stacking）

## 目的
アンサンブル手法の違い・長所短所を整理し、
**試験段階でも比較しやすい**形で運用方針を決める。

## 前提（試験段階）
- アンサンブルは **推論で常用しない前提でも leaderboard に載せて比較**する。
- 比較評価は `ensemble.selection_metric` を基準にする（既定は `eval.primary_metric` の holdout 指標）。
- CV 指標は `metrics.json` に残し、安定性チェックや stacking のメタ学習評価に使う。

## 手法別ポリシー
### mean_topk
- 概要: 上位 K モデルの予測を平均
- 長所: シンプル・再現性が高い・過学習リスクが低い
- 短所: モデル多様性が低い場合は改善幅が小さい
- 主要パラメータ: `ensemble.top_k`, `ensemble.selection_metric`

### weighted
- 概要: 重みを探索して加重平均（線形 search）
- 長所: mean_topk より改善する可能性がある
- 短所: 目的指標に対する **過適合**リスク、探索コストが増える
- 主要パラメータ: `ensemble.weighted.search`, `ensemble.weighted.n_samples`, `ensemble.weighted.seed`, `ensemble.weighted.top_k_max`

### stacking
- 概要: validation 予測を使ってメタモデルを学習（メタ側は CV で安定性評価）
- 長所: 非線形な組み合わせが可能、改善幅が大きいことがある
- 短所: **リークリスク**が最も高い。CV/分割の設計が必須。
- 主要パラメータ: `ensemble.stacking.meta_model`, `ensemble.stacking.cv_folds`, `ensemble.stacking.seed`, `ensemble.stacking.require_test_split`

## 推奨の比較ルール
- `ensemble.selection_metric` は原則 `eval.primary_metric` を使う。
- スコアが近い場合は **安定性（再現性/分散）と説明性**を優先する。
- stacking を使う場合は **CV で OOF を作る**ことを前提にする。

## どこを変更するか
- 既定の方針: `conf/ensemble/base.yaml`
- weighted/stacking のパラメータ: `conf/ensemble/weighted.yaml` / `conf/ensemble/stacking.yaml`
- アンサンブル有効化: `ensemble.enabled=true` または `task=train_ensemble`
- 実装ロジック: `src/tabular_analysis/processes/train_ensemble.py`
- 入出力仕様: `docs/05_PROCESS_CATALOG.md`

## leaderboard で比較する意義
- 単体モデルだけでは見えない上限性能を把握できる。
- 将来の「採用候補」を評価するための **比較ログ**として残せる。
- ただし **推薦（recommend）は自動、採用は推論時にユーザーが選択**を維持する。
