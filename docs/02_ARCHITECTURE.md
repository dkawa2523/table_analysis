# 02 Architecture

## 目的

このドキュメントは、solution repo の構造を最短で理解するための概要です。  
README が利用者向けの入口なら、このファイルは実装者向けの地図です。

## アーキテクチャの考え方

この solution は次の 3 層で考えると追いやすくなります。

### 1. 設定層

- Hydra config で task、model、preprocess、ClearML、execution policy を組み立てる
- 主な場所:
  - `conf/task/`
  - `conf/group/model/`
  - `conf/group/preprocess/`
  - `conf/clearml/`
  - `conf/exec_policy/`

### 2. 実行層

- `src/tabular_analysis/cli.py` が task を選び、`processes/` の各 task を起動する
- 各 task は `lifecycle.py` を通じて `config_resolved.yaml`、`out.json`、`manifest.json` を揃える

### 3. 運用層

- ClearML UI 上の project、template、pipeline controller、task metadata を整える
- 正本は visible pipeline template
- operator は Pipelines タブから template を確認、clone、実行する

## Polyrepo の役割分担

### `ml_platform`

- 共通 runtime
- Hydra / ClearML の基盤
- artifact / manifest / adapter の共通処理

### `ml-solution-tabular-analysis`

- tabular ドメインの task 実装
- model registry
- feature engineering
- reporting / evaluation
- ClearML UI 契約と pipeline 運用

## 標準フロー

```text
dataset_register
  -> preprocess
  -> train_model
  -> train_ensemble (optional)
  -> leaderboard
  -> infer (optional)
```

### dataset_register

- 入力データを raw dataset として登録
- schema と preview を確定

### preprocess

- split、feature type 判定、前処理 recipe を作成
- 再利用可能な preprocess bundle を出力

### train_model

- 単体モデル学習
- metrics、prediction、model bundle を出力

### train_ensemble

- train 済み候補から ensemble を構築
- 3 種の ensemble method を選択可能

### leaderboard

- 候補を比較し、推奨モデルを決める

### infer

- bundle path / train task / registry model から推論実行

### pipeline

- 上記をまとめて計画・実行
- `local/logging` ではローカル driver
- `pipeline_controller` では ClearML controller

## ClearML 側の見え方

### child template

- `dataset_register`
- `preprocess`
- `train_model`
- `train_ensemble`
- `leaderboard`
- `infer`

これらは strict template contract で lookup されます。

### visible pipeline template

- `pipeline`
- `train_model_full`
- `train_ensemble_full`

これらは `TaskTypes.controller` で作成され、Pipelines タブ上の正本になります。

project の canonical 形:

```text
<project_root>/TabularAnalysis/Pipelines
```

## queue 分割

現在の標準 queue:

- `services`
  - pipeline controller 専用
- `default`
  - preprocess、軽量学習、leaderboard、ensemble、infer
- `heavy-model`
  - `catboost`、`xgboost`

## bootstrap の考え方

- task ごとに独立環境で動かす
- venv の使い回しはしない
- ただし `uv` cache は共有する
- optional dependency は task 単位・model 単位で最小化する

例:

- `pipeline/preprocess/leaderboard` は base dependencies のみ
- `lgbm` は `lightgbm` のみ
- `xgboost` は `xgboost` のみ
- `catboost` は `catboost` のみ

## コードを追う順番

1. `src/tabular_analysis/cli.py`
2. `conf/task/<task>.yaml`
3. `src/tabular_analysis/processes/<task>.py`
4. 関連する registry / common / clearml module

### 例: pipeline を追いたい

1. `conf/task/pipeline.yaml`
2. `src/tabular_analysis/processes/pipeline.py`
3. `src/tabular_analysis/clearml/pipeline_templates.py`
4. `src/tabular_analysis/platform_adapter_pipeline.py`
5. `conf/clearml/templates.yaml`

### 例: model variant を増やしたい

1. `conf/group/model/<variant>.yaml`
2. `src/tabular_analysis/registry/models.py`
3. `conf/pipeline/model_sets/regression_all.yaml` など必要な model set

## 関連ドキュメント

- [05_PROCESS_CATALOG.md](05_PROCESS_CATALOG.md)
- [10_OPERATION_MODES.md](10_OPERATION_MODES.md)
- [60_PIPELINE_TRAIN_CONTRACT.md](60_PIPELINE_TRAIN_CONTRACT.md)
- [65_DEV_GUIDE_DIRECTORY_MAP.md](65_DEV_GUIDE_DIRECTORY_MAP.md)
- [81_CLEARML_TEMPLATE_POLICY.md](81_CLEARML_TEMPLATE_POLICY.md)
- [82_CLEARML_PROJECT_LAYOUT.md](82_CLEARML_PROJECT_LAYOUT.md)

