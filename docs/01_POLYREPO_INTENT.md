# 01_POLYREPO_INTENT（なぜポリレポか）

本プロジェクトは polyrepo 方針に従い、
**Platform（共通基盤）と Solution（用途別製品）を分離**します。

## Platform（ml-platform）の役割
- ClearML / Hydra / Artifacts の「契約」と共通ユーティリティ
- 実行モード（local / logging / agent / clone）
- manifest/out.json/hash の生成など、全 Solution に共通な追跡性
- registry interface（拡張点の型）

> 本 Solution は、Platform へ domain 固有ロジックを入れません（禁止）。

## Solution（ml-solution-tabular-analysis）の役割
- tabular 固有の前処理、モデル、評価、推論モード
- tabular の pipeline（どのタスクをどう繋ぐか）
- 非DS向けの運用手順（ClearML 上での見え方と判断導線）

## ポリレポのメリット
- ClearML 上で「どの用途のタスクか」が URL と Project 階層で一瞬で判別できる
- 変更影響範囲が小さい（レビュー・CI・リリースが軽い）
- Solution が増えても兄弟 repo を増やすだけでスケールできる
