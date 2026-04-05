# 84 Rehearsal Guide

## 目的

rehearsal は、最小の toy dataset を使って local から ClearML 運用までを一通り確認するための手順です。

## canonical tool

- `tools/rehearsal/run_pipeline_v2.py`

この runner は:

- toy dataset を作る
- `dataset_register` で `raw_dataset_id` を作り、その後に fixed-DAG の `pipeline` を回す
- `usecase_id`, `raw_dataset_id`, `pipeline_task_id` をまとめる

## 推奨フロー

### 1. local

```bash
python tools/rehearsal/run_pipeline_v2.py \
  --execution local \
  --task-type regression \
  --preprocess stdscaler_ohe \
  --models ridge,elasticnet
```

### 2. logging

```bash
python tools/rehearsal/run_pipeline_v2.py \
  --execution logging \
  --task-type regression \
  --preprocess stdscaler_ohe \
  --models ridge,elasticnet \
  --project-root LOCAL
```

### 3. pipeline controller

```bash
python tools/clearml_templates/manage_templates.py --apply --project-root LOCAL
python tools/clearml_templates/manage_templates.py --validate --project-root LOCAL

python tools/rehearsal/run_pipeline_v2.py \
  --execution agent \
  --queue-name controller \
  --task-type regression \
  --preprocess stdscaler_ohe \
  --models ridge,elasticnet \
  --project-root LOCAL \
  --skip-ui-verify
```

## よく見る出力

- `work/rehearsal/out/<mode>/<usecase_id>/rehearsal_summary.json`
- `99_pipeline/report.json`
- `99_pipeline/report_links.json`

## UI 監査

```bash
python tools/tests/rehearsal_verify_clearml_ui.py --usecase-id <USECASE_ID> --project-root LOCAL
```


