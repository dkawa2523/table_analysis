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

seed pipeline の標準運用では、operator は主に次を編集します。

- `run.usecase_id`
- `data.raw_dataset_id`
- `pipeline.selection.enabled_preprocess_variants`
- `pipeline.selection.enabled_model_variants`
- `ensemble.selection.enabled_methods`
- `ensemble.top_k` (`train_ensemble_full` のみ)

確認場所の優先順は次です。

- `Configuration > OperatorInputs`
  - operator が見るべき最小入力だけを mirror した read-only 表示
  - seed clone 実行前は `data.raw_dataset_id` が placeholder かどうかをここで確認する
- `Hyperparameters`
  - 実行ソースの正本
  - 互換 key や低レベル override もここに残る
  - `data.raw_dataset_id` を実際に差し替えるときもこちらを使う

seed pipeline の operator UI では、graph-shaping key を直接編集しません。

- `pipeline.model_variants`
- `pipeline.grid.model_variants`
- `pipeline.hpo.*`

これらは local / ad hoc 実行や開発者向け config の互換用として残し、通常の clone / run 導線では `selection` section を使います。

### clearml

- `run.clearml.enabled`
- `run.clearml.execution`
- `run.clearml.project_root`
- `run.clearml.code_ref.*`


