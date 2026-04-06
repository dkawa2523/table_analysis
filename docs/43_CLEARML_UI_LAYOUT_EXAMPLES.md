# 43 ClearML UI Layout Examples

## 目的

このファイルは、ClearML 上で task がどのように並ぶべきかを、軽量な例で確認するためのものです。

## 代表的な project layout

```text
LOCAL/TabularAnalysis/.pipelines/pipeline
LOCAL/TabularAnalysis/.pipelines/train_model_full
LOCAL/TabularAnalysis/.pipelines/train_ensemble_full
LOCAL/TabularAnalysis/Pipelines/Runs/<usecase_id>
LOCAL/TabularAnalysis/Runs/<usecase_id>/01_Datasets
LOCAL/TabularAnalysis/Runs/<usecase_id>/02_Preprocess
LOCAL/TabularAnalysis/Runs/<usecase_id>/03_TrainModels
LOCAL/TabularAnalysis/Runs/<usecase_id>/04_Ensembles
LOCAL/TabularAnalysis/Runs/<usecase_id>/05_Infer
LOCAL/TabularAnalysis/Runs/<usecase_id>/05_Infer_Children
LOCAL/TabularAnalysis/Runs/<usecase_id>/99_Leaderboard
```

## 見え方の期待

- seed pipeline は `.pipelines/<profile>` 配下、run controller は `Pipelines/Runs/<usecase_id>` 配下に見える
- child task は `Runs/<usecase_id>` 配下の project tree に整理される
- preprocess と train が混ざらない
- infer child は親 infer と別 group に分かれる
- seed clone の `run.usecase_id` を `TabularAnalysis` のまま起動した場合も、actual run では runtime が一意な `<usecase_id>` を採番する

## operator が見る順番

1. `.pipelines/<profile>`
2. `Pipelines/Runs/<usecase_id>`
3. 対象 usecase の `01_Datasets`
4. `02_Preprocess`
5. `03_TrainModels`
6. `04_Ensembles`
7. `05_Infer`


