# 61 ClearML HyperParameters Sections

## 目的

HyperParameters を section ごとに分け、operator が UI で設定の要点を読みやすくするための整理です。

## 既定 section

- `inputs`
- `dataset`
- `selection`
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

- `data.dataset_path`
- `data.target_column`
- `data.raw_dataset_id`
- `data.processed_dataset_id`

### selection

- `pipeline.selection.enabled_preprocess_variants`
- `pipeline.selection.enabled_model_variants`
- `ensemble.selection.enabled_methods`

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

- `pipeline.profile`
- `pipeline.run_*`
- `pipeline.plan_only`
- `pipeline.model_set`

visible pipeline template の operator UI では、graph-shaping key を直接編集しません。

- `pipeline.model_variants`
- `pipeline.grid.model_variants`
- `pipeline.hpo.*`

これらは local / ad hoc 実行や開発者向け config の互換用として残し、通常の clone / run 導線では `selection` section を使います。

### clearml

- `run.clearml.enabled`
- `run.clearml.execution`
- `run.clearml.project_root`
- `run.clearml.code_ref.*`


