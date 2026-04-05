# 03_CLEARML_UI_CONTRACT（ClearML UI 契約）

このドキュメントは **非DSユーザーが ClearML UI だけで判断できる**ための契約です。

## Project 階層（config-driven）

operator が主に見る階層:

- pipeline seed: `<ROOT>/<solution_root>/.pipelines/<profile>`
- pipeline run: `<ROOT>/<solution_root>/Pipelines/Runs/<usecase_id>`
- child task run: `<ROOT>/<solution_root>/Runs/<usecase_id>/<process_group>`
- step template: `<ROOT>/<solution_root>/Templates/Steps/<process_group>`

- ROOT: `run.clearml.project_root`（または `TABULAR_ANALYSIS_CLEARML_PROJECT_ROOT`）
- solution_root: `run.clearml.project_layout.solution_root`（例: `TabularAnalysis`）
- usecase_id: `run.usecase_id`（未指定なら `run.usecase_id_policy` で自動生成）
- process_group: `run.clearml.project_layout.group_map[process]`（未定義は `run.clearml.project_layout.misc_group`）
- 設定ファイル: `conf/clearml/project_layout.yaml`

例（デフォルト）：
- `MFG/TabularAnalysis/.pipelines/pipeline`
- `MFG/TabularAnalysis/Pipelines/Runs/test_toy_20260101_120000`
- `MFG/TabularAnalysis/Templates/Steps/03_TrainModels`
- `MFG/TabularAnalysis/Runs/test_toy_20260101_120000/01_Datasets`
- `MFG/TabularAnalysis/Runs/test_toy_20260101_120000/02_Preprocess`
- `MFG/TabularAnalysis/Runs/test_toy_20260101_120000/03_TrainModels`
- `MFG/TabularAnalysis/Runs/test_toy_20260101_120000/04_Ensembles`
- `MFG/TabularAnalysis/Runs/test_toy_20260101_120000/05_Infer`
- `MFG/TabularAnalysis/Runs/test_toy_20260101_120000/05_Infer_Children`（batch/optimize の child）
- `MFG/TabularAnalysis/Runs/test_toy_20260101_120000/99_Leaderboard`

## Task 名（推奨）
`<process>__<variant>__v<schema_version>`

例：
- `train_model__lgbm__preprocess=stdscaler_ohe__v1`

## Tags（最低限、固定キー）
- `usecase:<usecase_id>`
- `process:<process>`
- `schema:<schema_version>`
- `grid:<grid_run_id>`（pipeline 実行時）
- skip 時は `skipped:true`, `skip_reason:<reason>` を追加する

追加タグは `run.clearml.policy.tags` / `run.clearml.extra_tags` で付与できるが、上記キーは必須。
## User Properties（固定キー）
- `usecase_id`
- `process`
- `schema_version`
- `code_version`
- `platform_version`
- `grid_run_id`

追加の user properties は `run.clearml.policy.properties` で付与できるが、上記キーは必須。
追加（プロセス別）例：
- preprocess: `processed_dataset_id`, `split_hash`, `recipe_hash`
- train_model: `processed_dataset_id`, `split_hash`, `model_id`, `primary_metric`, `best_score`, `task_type`, `n_classes`
- train_ensemble: `processed_dataset_id`, `split_hash`, `model_id`, `primary_metric`, `best_score`, `task_type`, `n_classes`
- leaderboard: `recommended_train_task_id`, `recommended_model_id`, `excluded_count`

## HyperParameters（汚染防止：重要）
- **そのタスクの再現に必要な入力のみ**を記録する
- pipeline の設定や出力値を train の HyperParameters に混ぜない

seed pipeline について:

- `.pipelines/<profile>` にある seed pipeline は profile 固定の DAG です
- operator が UI から編集してよいのは `run.usecase_id`, `data.raw_dataset_id`, `pipeline.selection.enabled_preprocess_variants`, `pipeline.selection.enabled_model_variants`、必要時のみ `ensemble.selection.enabled_methods`, `ensemble.top_k` などの限定された項目だけです
- `pipeline.model_set` や `pipeline.grid.model_variants` は fixed profile を支える内部の graph-shaping 値として扱い、通常の clone / run 導線では触りません
- seed pipeline の標準運用は `pipeline.run_dataset_register=false` 前提で、dataset 登録は rehearsal / 準備系導線に分けます

## Artifacts（全タスク必須）
- `config_resolved.yaml`
- `out.json`
- `manifest.json`

プロセス別追加（例）：
- preprocess: `recipe.json`, `summary.md`, `preprocess_bundle.*`, `schema.json`
- train_model: `metrics.json`, `metrics_ci.json` (when `eval.ci.enabled=true`), `model_bundle/*`, `model_card.md`, `feature_importance.csv`, `feature_importance.png`, `residuals.png`, `confusion_matrix.csv`, `confusion_matrix.png`, `roc_curve.png`
- train_ensemble: `metrics.json`, `ensemble_spec.json`, `model_bundle.joblib`
- leaderboard: `leaderboard.csv`, `recommendation.json`, `summary.md`, `decision_summary.md`, `decision_summary.json`, `recommended_plot.png` (optional)
- pipeline: `pipeline_run.json`, `plan.json`, `report.md`, `report.json`, `report_links.json`, `run_summary.json`
- infer: `predictions.*`, `input_preview.*`, `drift_report.json`, `drift_report.md` (when drift enabled)
- skip 時: `skip_reason.json`

## Lint ルール（doctor/CI）
- 必須 artifact: `config_resolved.yaml`, `out.json`, `manifest.json`
- `manifest.json` 必須キー: `schema_version`, `code_version`, `platform_version`, `process`, `created_at`, `inputs`, `outputs`, `hashes.config_hash`
- `out.json` 必須キー（プロセス別）
  - dataset_register: `raw_dataset_id`
  - preprocess: `processed_dataset_id`, `split_hash`, `recipe_hash`
  - train_model: `model_id`, `primary_metric`, `best_score`, `task_type`
  - train_ensemble: `model_id`, `primary_metric`, `best_score`, `task_type`
  - leaderboard: `leaderboard_csv`, `recommended_model_id`
  - infer: `predictions_path`
  - pipeline: `pipeline_run`
- skip 時の `out.json` は `status="skipped"` と `reason` を必須で含める
- 任意 artifact は **存在すれば整形チェックのみ**（JSON は parse、MD は空でないこと）
- 実行例: `python -m tabular_analysis.doctor --lint-run <output_dir> --mode fail`

## Plots（軽量デフォルト）
- デフォルトは軽い図（重要度・残差など）
- 例: `feature_importance.png`, `residuals.png`, `confusion_matrix.png`, `roc_curve.png`
- SHAP 等の重い可視化は config フラグでオンデマンド（デフォルトOFF）

