# 01 Polyrepo Intent

## 背景

このプロジェクトは polyrepo 前提で構成されています。  
理由は、platform の汎用責務と solution の domain 責務を分け、solution ごとの進化を速くするためです。

## repository の分担

### `ml_platform`

platform repo は次を担当します。

- 共通 runtime
- Hydra / ClearML の基盤
- artifact / manifest / hashing の共通処理
- 汎用 adapter や utility

### `ml-solution-tabular-analysis`

solution repo は次を担当します。

- tabular ドメインの task 実装
- tabular 向け前処理
- model registry と model set
- leaderboard / infer / reporting
- ClearML UI 契約と operator 運用

## なぜ mono-repo にしないのか

polyrepo にしている理由:

- platform の release と solution の改善を分けたい
- domain 固有ロジックを solution 側で素早く変更したい
- ClearML 運用の UI 契約を solution 側で閉じたい

## 実装上の原則

### platform API への依存は adapter family 経由

solution の process や registry から platform を直接広く参照しないようにしています。  
接続面は `platform_adapter_*.py` に寄せています。

### platform へ戻すべきものは戻す

tabular 固有でない helper が増えたら platform へ戻す候補です。  
ただし、solution でしか意味がない UI 契約や運用ロジックは solution 側に残します。

### docs も polyrepo 前提

platform の概念説明は platform 側、tabular 固有の挙動は solution 側の docs に置きます。

## この repo で読むべき場所

- solution の入口: `README.md`
- code の地図: [65_DEV_GUIDE_DIRECTORY_MAP.md](65_DEV_GUIDE_DIRECTORY_MAP.md)
- platform との接点: [13_PLATFORM_INTEGRATION.md](13_PLATFORM_INTEGRATION.md)


