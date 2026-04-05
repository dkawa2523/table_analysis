# 43 ClearML UI Layout Examples

## 目的

このファイルは、ClearML 上で task がどのように並ぶべきかを、軽量な例で確認するためのものです。

## 代表的な project layout

```text
LOCAL/TabularAnalysis/.pipelines/pipeline
LOCAL/TabularAnalysis/.pipelines/train_model_full
LOCAL/TabularAnalysis/.pipelines/train_ensemble_full
LOCAL/TabularAnalysis/Pipelines/Runs/<usecase_id>
LOCAL/TabularAnalysis/<usecase_id>/01_Datasets
LOCAL/TabularAnalysis/<usecase_id>/02_Preprocess
LOCAL/TabularAnalysis/<usecase_id>/03_TrainModels
LOCAL/TabularAnalysis/<usecase_id>/04_Ensembles
LOCAL/TabularAnalysis/<usecase_id>/05_Infer
LOCAL/TabularAnalysis/<usecase_id>/05_Infer_Children
```

## 見え方の期待

- seed pipeline は `.pipelines/<profile>` 配下、run controller は `Pipelines/Runs/<usecase_id>` 配下に見える
- child task は usecase ごとの project tree に整理される
- preprocess と train が混ざらない
- infer child は親 infer と別 group に分かれる

## operator が見る順番

1. `.pipelines/<profile>`
2. `Pipelines/Runs/<usecase_id>`
3. 対象 usecase の `01_Datasets`
4. `02_Preprocess`
5. `03_TrainModels`
6. `04_Ensembles`
7. `05_Infer`


