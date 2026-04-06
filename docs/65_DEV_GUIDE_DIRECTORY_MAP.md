# Dev Guide Directory Map

主要ディレクトリの責務を、ClearML pipeline 運用の観点でまとめる。

## Source

- `src/tabular_analysis/processes`
  - 各 task の実処理
  - `pipeline.py`, `leaderboard.py`, `infer.py`, `train_model.py` など
- `src/tabular_analysis/processes/*_support.py`
  - orchestration から分離した純粋 helper
- `src/tabular_analysis/clearml`
  - Hyperparameters section, seed lookup, UI 契約
- `src/tabular_analysis/ops`
  - project path, naming, identity, runtime env
- `src/tabular_analysis/viz`
  - leaderboard などの PLOTS 生成

## Config

- `conf/task`
  - task ごとの Hydra config
- `conf/clearml/templates.yaml`
  - step template / seed template 定義
- `conf/clearml/templates.lock.yaml`
  - live task id の lock
- `conf/clearml/project_layout.yaml`
  - ClearML project path の正本

## Tools

- `tools/clearml_templates/manage_templates.py`
  - seed / step template の apply / validate / cleanup
- `tools/clearml_entrypoint.py`
  - ClearML agent 起動時の override 正規化
- `tools/tests`
  - contract, smoke, lint, UI 回帰

## Docs

- `docs/17_OPERATOR_QUICKSTART.md`
  - operator 向け最短手順
- `docs/52_CLEARML_PIPELINE_CONTROLLER_CONTRACT.md`
  - pipeline controller 契約
- `docs/61_CLEARML_HPARAMS_SECTIONS.md`
  - Hyperparameters section の意味
- `docs/87_CLEARML_PIPELINE_WORKFLOW_DETAILS.md`
  - seed から infer までの全体ワークフロー
