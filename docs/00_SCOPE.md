# 00 Scope

## この solution の対象

`ml-solution-tabular-analysis` は、tabular データを対象にした機械学習 solution repo です。  
この repo は「表形式データの前処理、学習、比較、推論、ClearML 運用」を一貫して扱うことを目的にしています。

## In Scope

この repo が責任を持つ範囲:

- CSV / Parquet を入力にした tabular workflow
- 回帰 / 二値分類の tabular モデル学習
- 前処理 recipe と split の再現
- 単体モデルと ensemble の比較評価
- leaderboard による推奨モデル選定
- ClearML による task / artifact / pipeline / model traceability
- ClearML Pipelines タブを正本とした pipeline template 運用
- task-time install 前提の再現可能 runtime

## Out Of Scope

この repo の責務ではないもの:

- 画像、音声、NLP など tabular 以外の domain
- platform 基盤そのものの設計変更
- 大規模な feature store / online serving 基盤
- notebook ベースの探索作業の管理

## 重要な設計方針

### 1. task ごとの独立実行

各 task は単体でも実行できることを重視します。

- `dataset_register`
- `preprocess`
- `train_model`
- `train_ensemble`
- `leaderboard`
- `infer`
- `pipeline`

### 2. 一括実行と単体実行の両立

- operator は `pipeline_controller` で一括実行できる
- 開発者は individual task を単体で確認できる

### 3. traceability を失わない

各 task は少なくとも次を残します。

- `config_resolved.yaml`
- `out.json`
- `manifest.json`

ClearML を使う場合は、project、tags、properties、artifacts、HyperParameters でも追跡できることを重視します。

### 4. comparability を明示する

モデル比較は「同じ split、同じ recipe、同じ評価条件」で行う前提です。  
`split_hash` と `recipe_hash` を比較可能性の中心に据えています。

## この repo の主な利用者

- operator
  - ClearML UI から template を確認し、clone して pipeline を実行する
- ML engineer
  - 前処理、モデル、評価、report を実装・改善する
- reviewer / newcomer
  - docs と task 出力から current behavior を理解する

## 関連ドキュメント

- [01_POLYREPO_INTENT.md](01_POLYREPO_INTENT.md)
- [02_ARCHITECTURE.md](02_ARCHITECTURE.md)
- [05_PROCESS_CATALOG.md](05_PROCESS_CATALOG.md)
- [10_OPERATION_MODES.md](10_OPERATION_MODES.md)

