# 61 ClearML HyperParameters Sections

## 目的

HyperParameters を section ごとに分け、operator が UI で設定の要点を読みやすくするための整理です。

## 既定 section

- `inputs`
- `dataset`
- `preprocess`
- `model`
- `eval`
- `optimize`
- `pipeline`
- `clearml`

## 代表例

### inputs

- `run.usecase_id`
- `run.output_dir`
- `data.dataset_path`
- `infer.mode`

### dataset

- `data.raw_dataset_id`
- `data.processed_dataset_id`

### preprocess

- `preprocess.*`
- `data.split.*`

### model

- `group/model`
- `train.*`

### eval

- `eval.*`
- `leaderboard.*`

### pipeline

- `pipeline.*`

### clearml

- `run.clearml.enabled`
- `run.clearml.execution`
- `run.clearml.project_root`
- `run.clearml.code_ref.*`


