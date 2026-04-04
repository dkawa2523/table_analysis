# 24 Reporting

## 目的

この repo の reporting は、operator と開発者が「何が起きたか」を task 出力だけで追えるようにすることを目的にしています。

## reporting の層

### task local output

各 task は stage directory に report 相当の情報を出します。

### ClearML UI

必要な scalars、plots、tables、artifacts を UI から見られるようにします。

### pipeline summary

pipeline は複数 task の結果を `report.md` / `report.json` に集約します。

## 代表的な report artifact

### preprocess

- `summary.md`
- `data_quality.json`
- `data_quality.md`

### train_model

- `metrics.json`
- plots / scalars

### leaderboard

- `leaderboard.csv`
- `recommendation.json`
- `summary.md`

### pipeline

- `pipeline_run.json`
- `run_summary.json`
- `report.md`
- `report.json`
- `report_links.json`

## report.md と report.json の使い分け

### `report.md`

- 人が読む要約
- operator 向け

### `report.json`

- 機械可読の summary
- automation / regression check 向け
- lifecycle の正本は `pipeline_run.json`
- `report.json` と `run_summary.json` はその状態を読みやすく投影したもの

## pipeline report で必ず見たい項目

- `grid_run_id`
- `status`
- `recommended_train_task_id`
- `infer_model_id`
- `primary_metric`
- `best_score`
- `planned_jobs`
- `executed_jobs`
- `completed_jobs`
- `failed_jobs`
- `stopped_jobs`
- `running_jobs`
- `queued_jobs`
- `skipped_due_to_policy`

## partial failure の扱い

pipeline は `pipeline_run.json` を lifecycle の canonical source とし、stopped / skipped / partial failure の情報も `run_summary.json` と `report.json` に残します。  
途中で止まっても「どこまで完了したか」が分かることを重視しています。

## 関連ドキュメント

- [24_REPORTING.md](24_REPORTING.md)
- [51_CLEARML_PLOTS_SCALARS_DEBUGSAMPLES_CONTRACT.md](51_CLEARML_PLOTS_SCALARS_DEBUGSAMPLES_CONTRACT.md)
- [60_PIPELINE_TRAIN_CONTRACT.md](60_PIPELINE_TRAIN_CONTRACT.md)


