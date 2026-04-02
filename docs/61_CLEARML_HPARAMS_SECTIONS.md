# 61 ClearML HyperParameters Sections

## Goal
ClearML HyperParameters should be grouped into stable sections instead of
falling back to a single General bucket.

## How it works
- Section extraction lives in `src/tabular_analysis/clearml/hparams.py`
- Section connections are sent through `src/tabular_analysis/platform_adapter_task.py`
- Section definitions come from `conf/clearml/hyperparams_sections.yaml` or `run.clearml.hyperparams.sections`

## Default sections
- `inputs`
- `dataset`
- `preprocess`
- `model`
- `eval`
- `optimize`
- `pipeline`
- `clearml`

## Example dotpaths
- `inputs`: `run.usecase_id`, `run.output_dir`, `data.dataset_path`, `infer.mode`
- `dataset`: `data.raw_dataset_id`, `data.processed_dataset_id`
- `preprocess`: `preprocess.*`, `data.split.*`, `ops.processed_dataset.*`
- `model`: `train.model`, `train.params`, `model_variant.*`
- `eval`: `eval.*`, `leaderboard.*`
- `pipeline`: `pipeline.*`
- `clearml`: `run.clearml.enabled`, `run.clearml.execution`, `run.clearml.code_ref.*`
