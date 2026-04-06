# ClearML Minimality Guide

このドキュメントは、ClearML 連携で複雑さを増やさないための設計方針をまとめる。

## 基本原則

- UI 正本は seed pipeline のみ
- 実行正本は actual run のみ
- operator が編集するのは `Hyperparameters` のみ
- `Configuration > OperatorInputs` は確認用 mirror のみ
- pipeline 固有の特別扱いは増やさず、step template と同じ task 契約を優先する

## 避けること

- `NEW PIPELINE` 前提の UI authoring
- seed と template の dual-write
- old project namespace を lookup 対象に残すこと
- `%2E` のような encoded key を実行経路に流すこと

## 現在の正本

- seed: `<project_root>/TabularAnalysis/.pipelines/<profile>`
- controller run: `<project_root>/TabularAnalysis/Pipelines/Runs/<usecase_id>`
- child task: `<project_root>/TabularAnalysis/Runs/<usecase_id>/<group>`

## 実装上の最小責務

- seed task は `NEW RUN` の起点
- controller は DAG と child orchestration
- leaderboard は推薦と infer 参照の決定
- infer は standalone task として実行

関連:

- [03_CLEARML_UI_CONTRACT.md](03_CLEARML_UI_CONTRACT.md)
- [52_CLEARML_PIPELINE_CONTROLLER_CONTRACT.md](52_CLEARML_PIPELINE_CONTROLLER_CONTRACT.md)
- [81_CLEARML_TEMPLATE_POLICY.md](81_CLEARML_TEMPLATE_POLICY.md)
