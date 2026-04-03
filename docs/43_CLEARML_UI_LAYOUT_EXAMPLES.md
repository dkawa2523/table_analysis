# 43 ClearML UI Layout Examples

## 目的

このファイルは、ClearML 上で task がどのように並ぶべきかを、軽量な例で確認するためのものです。

## 代表的な project layout

```text
LOCAL/TabularAnalysis/Pipelines
LOCAL/TabularAnalysis/<usecase_id>/01_Datasets
LOCAL/TabularAnalysis/<usecase_id>/02_Preprocess
LOCAL/TabularAnalysis/<usecase_id>/03_TrainModels
LOCAL/TabularAnalysis/<usecase_id>/04_Ensembles
LOCAL/TabularAnalysis/<usecase_id>/05_Infer
LOCAL/TabularAnalysis/<usecase_id>/05_Infer_Children
```

## 見え方の期待

- pipeline template と run controller は `Pipelines` 配下に見える
- child task は usecase ごとの project tree に整理される
- preprocess と train が混ざらない
- infer child は親 infer と別 group に分かれる

## operator が見る順番

1. `Pipelines`
2. 対象 usecase の `01_Datasets`
3. `02_Preprocess`
4. `03_TrainModels`
5. `04_Ensembles`
6. `05_Infer`


