# 17 Operator Quickstart

この 1 ページは、operator が ClearML UI から seed pipeline を開き、`NEW RUN` で安全に実行するための最短手順です。  
詳細な運用手順は [16_OPERATIONS_RUNBOOK.md](16_OPERATIONS_RUNBOOK.md)、問題発生時は [69_CLEARML_TROUBLESHOOTING.md](69_CLEARML_TROUBLESHOOTING.md) を参照してください。

## 1. まず何を使うか

現在の標準入口は seed pipeline card です。

- `LOCAL/TabularAnalysis/.pipelines/pipeline`
- `LOCAL/TabularAnalysis/.pipelines/train_model_full`
- `LOCAL/TabularAnalysis/.pipelines/train_ensemble_full`

使い分けは次です。

| seed profile | 何を実行するか | 典型用途 |
| --- | --- | --- |
| `pipeline` | preprocess + single-model train + leaderboard | 標準の学習実行 |
| `train_model_full` | preprocess + single-model train | 単体モデル群だけを見たいとき |
| `train_ensemble_full` | preprocess + single-model train + 3 ensemble + leaderboard | フル構成の比較・採用判断 |

## 2. 事前確認

```bash
python tools/clearml_templates/manage_templates.py --apply --project-root LOCAL
python tools/clearml_templates/manage_templates.py --validate --project-root LOCAL
```

確認したいこと:

- seed card が `.pipelines/<profile>` に見える
- `controller`, `default`, `heavy-model` queue に worker がいる
- 実行したい raw dataset id が分かっている

## 3. UI での標準手順

1. ClearML UI で目的の seed card を開く
2. `NEW RUN` を押す
3. まず `Configuration > OperatorInputs` を見る
4. 次に `Hyperparameters` を開く
5. `data.raw_dataset_id` を実在する raw dataset id に差し替える
6. 必要なら `run.usecase_id` や selection を更新する
7. 実行する

## 4. どこで何を確認・編集するか

| 画面 | 役割 | 編集可否 | 主に見る項目 |
| --- | --- | --- | --- |
| `Configuration > OperatorInputs` | 確認用 mirror | しない | `run.usecase_id`, `data.raw_dataset_id`, `pipeline.selection.*`, `ensemble.selection.*` |
| `Hyperparameters` | 実行ソースの正本 | する | `run.usecase_id`, `data.raw_dataset_id`, `pipeline.selection.*`, `ensemble.selection.*`, `ensemble.top_k` |

重要:

- seed card の `data.raw_dataset_id=REPLACE_WITH_EXISTING_RAW_DATASET_ID` は正常です
- 実際の編集は `Hyperparameters` 側で行います
- placeholder のまま actual run を開始すると fail-fast します

## 5. 通常編集してよい項目

- `run.usecase_id`
- `data.raw_dataset_id`
- `pipeline.selection.enabled_preprocess_variants`
- `pipeline.selection.enabled_model_variants`
- `ensemble.selection.enabled_methods`
- `ensemble.top_k`

通常は編集しない項目:

- `pipeline.model_set`
- `pipeline.grid.model_variants`
- `pipeline.hpo.*`
- `run.clearml.pipeline.template_task_id`
- `run.clearml.template_usecase_id`

上の項目は開発者向けの互換・内部設定です。operator の通常運用では触りません。

## 6. `run.usecase_id` の扱い

`run.usecase_id` を明示的に決めたい場合は、自分で分かりやすい値を入れてください。  
未編集で seed 既定値 `TabularAnalysis` のまま実行しても、actual run では runtime が一意値へ自動採番します。

現在の標準 policy は `dataset_timestamp` で、例は次です。

```text
test_e285ff784b9046b7b1f9920e54e3fe93_20260405_140419
```

## 7. 実行後に見る場所

run controller:

- `LOCAL/TabularAnalysis/Pipelines/Runs/<usecase_id>`

child task:

- `LOCAL/TabularAnalysis/Runs/<usecase_id>/01_Datasets`
- `LOCAL/TabularAnalysis/Runs/<usecase_id>/02_Preprocess`
- `LOCAL/TabularAnalysis/Runs/<usecase_id>/03_TrainModels`
- `LOCAL/TabularAnalysis/Runs/<usecase_id>/04_Ensembles`
- `LOCAL/TabularAnalysis/Runs/<usecase_id>/05_Infer`
- `LOCAL/TabularAnalysis/Runs/<usecase_id>/05_Infer_Children`
- `LOCAL/TabularAnalysis/Runs/<usecase_id>/99_Leaderboard`

主な controller artifact:

- `pipeline_run.json`
- `run_summary.json`
- `report.json`
- `report.md`
- `report_links.json`

## 8. よくある見え方の補足

- seed card は `.pipelines/<profile>` に残り続けます
- actual run は `Pipelines/Runs/<usecase_id>` に出ます
- historical run には `%2E` を含む古い key 表示が残ることがあります
- current seed と新規 `NEW RUN` は plain dotted key を正本にしています

## 9. 関連ドキュメント

- [16_OPERATIONS_RUNBOOK.md](16_OPERATIONS_RUNBOOK.md)
- [55_CLEARML_UI_CHECKLIST.md](55_CLEARML_UI_CHECKLIST.md)
- [61_CLEARML_HPARAMS_SECTIONS.md](61_CLEARML_HPARAMS_SECTIONS.md)
- [69_CLEARML_TROUBLESHOOTING.md](69_CLEARML_TROUBLESHOOTING.md)
- [87_CLEARML_PIPELINE_WORKFLOW_DETAILS.md](87_CLEARML_PIPELINE_WORKFLOW_DETAILS.md)
