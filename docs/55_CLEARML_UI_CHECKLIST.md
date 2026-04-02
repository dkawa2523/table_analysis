# ClearML UI Checklist (Trial Phase, T097-aligned)

This checklist is a step-by-step guide to confirm ClearML UI outputs after running rehearsal or pipeline tasks.
See also: `docs/03_CLEARML_UI_CONTRACT.md`, `docs/50_CLEARML_PROCESSED_DATASET_CONTRACT.md`,
`docs/51_CLEARML_PLOTS_SCALARS_DEBUGSAMPLES_CONTRACT.md`, `docs/52_CLEARML_PIPELINE_CONTROLLER_CONTRACT.md`,
`docs/53_CLEARML_HYPERPARAMETERS_CONTRACT.md`, `docs/60_PIPELINE_TRAIN_CONTRACT.md`, `docs/83_ENSEMBLE_POLICY.md`.
(Reference: `docs/61_CLEARML_HPARAMS_SECTIONS.md`, `docs/63_CLEARML_PIPELINES_VISIBILITY.md`,
`docs/64_INFER_BATCH_OPTIMIZE_CONTRACT.md`, `docs/66_NAMING_TAGGING_POLICY.md`.)

Key assumptions (T097):
- Tags: `usecase:<usecase_id>` / `process:<process>` / `solution:tabular-analysis` / `schema:v1` / `grid:<grid_id>`
- Project hierarchy: `<project_root>/TabularAnalysis/<usecase_id>/<process_group>`
- HyperParameters sections: `inputs` / `dataset` / `preprocess` / `model` / `eval` / `pipeline` / `clearml`
- Local/Agent consistency: template clone + `run.clearml.code_ref.mode=branch` (no commit pin)

## 1. 事前準備（テンプレ Task の存在確認）
| 項目 | 見る場所 / 確認内容 |
| --- | --- |
| 探し方 | - [ ] Projects でテンプレ用 usecase（`run.clearml.template_usecase_id`）配下を開く（詳細は `conf/clearml/templates.yaml`）<br>- [ ] Tags で `template:true` / `template_set:<id>` / `process:<process>` / `solution:tabular-analysis` / `schema:v1` を確認 |
| Configuration | - [ ] テンプレは clone 前提のため、`run.clearml.code_ref.mode=branch` になっている（commit pin/version_num で落ちない）<br>- [ ] `run.clearml.execution=logging` で作成されている（Local/Agent の見え方一致） |
| Scalars | - [ ] テンプレ Task では必須スカラーなし |
| Plots | - [ ] テンプレ Task では必須プロットなし |
| Artifacts | - [ ] テンプレ Task の artifacts は参考程度（検索キーにしない） |
| Tags / User Properties | - [ ] Tags に `template:true` / `process:<process>` / `template_set:<id>` / `solution:tabular-analysis` / `schema:v1` を含む<br>- [ ] Properties に `usecase_id` / `process` / `schema_version` を含む |

## 2. dataset_register（raw dataset）
| 項目 | 見る場所 / 確認内容 |
| --- | --- |
| 探し方 | - [ ] Projects: `<project_root>/TabularAnalysis/<usecase_id>/01_Datasets`（推奨レイアウト）<br>- [ ] Tags: `usecase:<usecase_id>` / `process:dataset_register` / `schema:v1`（+ `solution:tabular-analysis` が付く運用なら確認）<br>- [ ] Datasets タブに raw dataset があり、`<usecase_id>__raw__<filename>` の命名になっている<br>- [ ] Dataset tags は `usecase:<usecase_id>` / `process:dataset_register` / `schema:v1` を含む |
| Configuration | - [ ] HyperParameters > inputs に `data.dataset_path`, `data.target_column`, `data.id_columns`, `data.drop_columns` が載る（必要時）<br>- [ ] HyperParameters > clearml に `run.clearml.execution`, `run.clearml.project_root`, `run.clearml.queue_name` が載る |
| Scalars | - [ ] 必須スカラーなし（品質ゲート有効時のみ summary が出る） |
| Plots | - [ ] データ品質/プロファイルが PLOTS に出る（有効時。Artifacts だけで終わらない） |
| Artifacts | - [ ] `config_resolved.yaml`, `out.json`, `manifest.json` がある（Artifacts は検索キーではない）<br>- [ ] 任意: `schema.json`, `preview.csv`, `data_quality.json` |
| Tags / User Properties | - [ ] Task Tags: `usecase:<usecase_id>` / `process:dataset_register` / `schema:v1`（+ `solution:tabular-analysis`）<br>- [ ] Task Properties: `usecase_id`, `process`, `schema_version`, `code_version`, `platform_version` |

## 3. preprocess（processed dataset + 復元性）
| 項目 | 見る場所 / 確認内容 |
| --- | --- |
| 探し方 | - [ ] Projects: `<project_root>/TabularAnalysis/<usecase_id>/02_Preprocess`<br>- [ ] Tags: `usecase:<usecase_id>` / `process:preprocess` / `schema:v1` / `grid:<grid_id>`（pipeline 時）<br>- [ ] Datasets タブに `processed__{usecase_id}__{preprocess_variant}__{split_hash}__{schema_version}` がある<br>- [ ] Dataset tags は `usecase:<usecase_id>` / `process:preprocess` / `type:processed` / `schema:v1` を含む |
| Configuration | - [ ] HyperParameters > dataset に `data.raw_dataset_id` が載る<br>- [ ] HyperParameters > preprocess に `preprocess.variant`, `data.split.*`, `ops.processed_dataset.*` が載る<br>- [ ] HyperParameters > clearml に `run.clearml.execution` が載る |
| Scalars | - [ ] 必須スカラーなし（品質ゲート有効時は summary が出る） |
| Plots | - [ ] raw/processed のサンプルや品質プロファイルが PLOTS に出る（有効時） |
| Artifacts | - [ ] `config_resolved.yaml`, `out.json`, `manifest.json`（Artifacts は検索キーではない）<br>- [ ] `recipe.json`, `summary.md`, `preprocess_bundle.joblib`, `schema.json`<br>- [ ] Processed Dataset に `split.json`, `schema.json`, `recipe.json`, `preprocess_bundle.joblib`, `meta.json`（store_features=true なら `X.parquet`, `y.parquet`） |
| Tags / User Properties | - [ ] Tags: `usecase:<usecase_id>` / `process:preprocess` / `schema:v1` / `grid:<grid_id>`（+ `solution:tabular-analysis`）<br>- [ ] Properties: `processed_dataset_id`, `split_hash`, `recipe_hash`, `schema_hash` |

## 4. train_model（単体モデル）
| 項目 | 見る場所 / 確認内容 |
| --- | --- |
| 探し方 | - [ ] Projects: `<project_root>/TabularAnalysis/<usecase_id>/03_TrainModels`<br>- [ ] Tags: `usecase:<usecase_id>` / `process:train_model` / `schema:v1` / `grid:<grid_id>`<br>- [ ] 追加タグ: `model:<abbr>`, `preprocess:<variant>`, `dataset:<raw_dataset_id>` |
| Configuration | - [ ] HyperParameters > dataset に `data.processed_dataset_id` が載る<br>- [ ] HyperParameters > model に `train.model`, `train.params`, `model_variant.name`, `model_variant.params` が載る<br>- [ ] HyperParameters > eval に `eval.primary_metric`, `eval.direction`, `eval.task_type` が載る |
| Scalars | - [ ] `metrics/<primary_metric>` がある<br>- [ ] `metrics/secondary/*` がある（該当時） |
| Plots | - [ ] feature importance（bar）<br>- [ ] classification: confusion matrix / ROC curve<br>- [ ] regression: residual plot<br>- [ ] Plotly 優先（png artifact のみはNG） |
| Artifacts | - [ ] `config_resolved.yaml`, `out.json`, `manifest.json`（Artifacts は検索キーではない）<br>- [ ] `metrics.json`（`metrics_ci.json` は CI 有効時）<br>- [ ] `model_bundle/*`, `feature_importance.csv/png`, `residuals.png`, `confusion_matrix.*`, `roc_curve.png` |
| Tags / User Properties | - [ ] Tags: `usecase:<usecase_id>` / `process:train_model` / `schema:v1` / `grid:<grid_id>`（+ `solution:tabular-analysis`）<br>- [ ] Properties: `processed_dataset_id`, `model_id`, `primary_metric`, `best_score`, `task_type`, `n_classes` |

## 5. ensemble（mean_topk / weighted / stacking）
| 項目 | 見る場所 / 確認内容 |
| --- | --- |
| 探し方 | - [ ] Projects: `<project_root>/TabularAnalysis/<usecase_id>/04_Ensembles`<br>- [ ] Tags: `process:train_ensemble` / `ensemble:mean_topk` / `ensemble:weighted` / `ensemble:stacking` / `topk:<k>`<br>- [ ] Task 名: `train_ensemble/<method>(k=<k>)` が 3 方式分ある（有効時） |
| Configuration | - [ ] HyperParameters > preprocess に `preprocess.variant` が載る<br>- [ ] HyperParameters > dataset に `data.processed_dataset_id` が載る<br>- [ ] HyperParameters > eval に `eval.primary_metric`, `eval.direction` が載る<br>- [ ] `ensemble.*` の詳細は `config_resolved.yaml` で確認（Artifacts） |
| Scalars | - [ ] `metrics/<primary_metric>` がある<br>- [ ] `ensemble/best_score`, `ensemble/n_included`, `ensemble/n_skipped` がある |
| Plots | - [ ] Tables (Plots): `included_models`, `skipped_reasons` が見える<br>- [ ] weighted: `weights` テーブルがある<br>- [ ] stacking: `meta_coefficients` テーブルがある |
| Artifacts | - [ ] `config_resolved.yaml`, `out.json`, `manifest.json`（Artifacts は検索キーではない）<br>- [ ] `metrics.json`, `ensemble_spec.json`, `model_bundle.joblib`（stacking は `meta_model.joblib`） |
| Tags / User Properties | - [ ] Tags: `usecase:<usecase_id>` / `process:train_ensemble` / `schema:v1` / `grid:<grid_id>`（+ `solution:tabular-analysis`）<br>- [ ] 方式タグ: `ensemble:<method>` / `topk:<k>` / `preprocess:<variant>`<br>- [ ] Properties: `processed_dataset_id`, `model_id`, `primary_metric`, `best_score`, `task_type` |

## 6. leaderboard（比較評価 + 推奨/採用の分離）
| 項目 | 見る場所 / 確認内容 |
| --- | --- |
| 探し方 | - [ ] Projects: `<project_root>/TabularAnalysis/<usecase_id>/00_Pipelines`（leaderboard もここに配置）<br>- [ ] Tags: `usecase:<usecase_id>` / `process:leaderboard` / `schema:v1` / `grid:<grid_id>` |
| Configuration | - [ ] HyperParameters > eval に `eval.primary_metric`, `eval.direction`, `leaderboard.require_comparable`, `leaderboard.top_k` が載る |
| Scalars | - [ ] `leaderboard/best_score` がある |
| Plots | - [ ] top-k bar（スコア比較） |
| Artifacts | - [ ] `config_resolved.yaml`, `out.json`, `manifest.json`（Artifacts は検索キーではない）<br>- [ ] `leaderboard.csv`, `recommendation.json`, `summary.md`, `decision_summary.md/json` |
| Tags / User Properties | - [ ] Tags: `usecase:<usecase_id>` / `process:leaderboard` / `schema:v1` / `grid:<grid_id>`（+ `solution:tabular-analysis`）<br>- [ ] Properties: `recommended_train_task_id`, `recommended_model_id`, `excluded_count` |

## 7. pipeline（PipelineController / 子タスク生成 / 階層化）
| 項目 | 見る場所 / 確認内容 |
| --- | --- |
| 探し方 | - [ ] Projects: `<project_root>/TabularAnalysis/<usecase_id>/00_Pipelines`<br>- [ ] Tags: `usecase:<usecase_id>` / `process:pipeline` / `schema:v1` / `grid:<grid_id>`<br>- [ ] ClearML UI の PIPELINES タブに controller が表示され、preprocess/train/ensemble/leaderboard（+ infer 有効時）がノードで見える |
| Configuration | - [ ] HyperParameters > pipeline に `pipeline.profile`, `pipeline.groups.*`, `pipeline.fail_policy.*`, `pipeline.limits.*` が載る<br>- [ ] HyperParameters > inputs に `run.grid_run_id` が載る<br>- [ ] `run.clearml.execution=pipeline_controller`（または logging/local の明示） |
| Scalars | - [ ] `pipeline/num_models`, `pipeline/num_succeeded`, `pipeline/num_failed` がある |
| Plots | - [ ] 必須プロットなし（report は artifacts を参照） |
| Artifacts | - [ ] `config_resolved.yaml`, `out.json`, `manifest.json`（Artifacts は検索キーではない）<br>- [ ] `pipeline_run.json`, `plan.json`, `report.md`, `report.json`, `report_links.json`, `run_summary.json` |
| Tags / User Properties | - [ ] Tags: `usecase:<usecase_id>` / `process:pipeline` / `schema:v1` / `grid:<grid_id>`（+ `solution:tabular-analysis`）<br>- [ ] 子タスクにも `usecase:<usecase_id>` が付与され、Project 階層に反映される |

※ Pipelines タブの表示には project system tag `pipeline` が必要なため、
`run.clearml.pipeline.project_tag_pipeline=true` を前提とする。

## 8. infer（single/batch/optimize の UI 表示）
| 項目 | 見る場所 / 確認内容 |
| --- | --- |
| 探し方 | - [ ] Projects: `<project_root>/TabularAnalysis/<usecase_id>/05_Infer`（summary）/ `<project_root>/TabularAnalysis/<usecase_id>/05_Infer_Children`（child）<br>- [ ] Tags: `usecase:<usecase_id>` / `process:infer` / `schema:v1`<br>- [ ] batch/optimize は summary task + child tasks（`infer__single__...`）がある |
| Configuration | - [ ] HyperParameters > inputs に `infer.mode`, `infer.input_path/json`, `infer.validation.mode`, `infer.batch.inputs_path/json` が載る<br>- [ ] HyperParameters > dataset に `data.raw_dataset_id` / `data.processed_dataset_id` が載る |
| Scalars | - [ ] `infer/latency_ms` がある（推奨） |
| Plots | - [ ] input vs output の例は Debug Samples ではなく PLOTS のテーブルで確認する<br>- [ ] batch: 全条件のテーブル + 予測分布などが summary に出る<br>- [ ] optimize: Optuna 可視化（history/parallel/importance/response surface）+ 上位条件テーブル |
| Artifacts | - [ ] `config_resolved.yaml`, `out.json`, `manifest.json`（Artifacts は検索キーではない）<br>- [ ] `predictions.*`, `input_preview.*`, `drift_report.json/md`（drift 有効時） |
| Tags / User Properties | - [ ] Tags: `usecase:<usecase_id>` / `process:infer` / `schema:v1`（+ `solution:tabular-analysis`）<br>- [ ] Properties: `drift_alert`（drift 有効時のみ） |

## 9. 失敗時の見方（SKIP/partial failure/optional deps）
| 項目 | 見る場所 / 確認内容 |
| --- | --- |
| 探し方 | - [ ] Tags で `skipped:true` / `skip_reason:<reason>` を検索<br>- [ ] PIPELINES タブの controller で failed/skipped ノードを確認 |
| Configuration | - [ ] HyperParameters で該当 variant / optional deps を確認（`preprocess.variant`, `model_variant.*` など） |
| Scalars | - [ ] pipeline の `pipeline/num_failed` / `pipeline/num_succeeded` を確認<br>- [ ] ensemble の `ensemble/n_skipped` を確認 |
| Plots | - [ ] ensemble の `skipped_reasons` テーブルを確認 |
| Artifacts | - [ ] `out.json` に `status=skipped/failed` と `reason/error` がある<br>- [ ] `skip_reason.json`（存在時）<br>- [ ] pipeline の `run_summary.json` で `degraded=true` / `status` / `policy.result.*` を確認<br>- [ ] `plan.json` の `groups.*.skipped_missing_dependencies` / `skipped_inapplicable` を確認 |
| Tags / User Properties | - [ ] Tags: `skipped:true`, `skip_reason:<reason>` が付与されている |

## 10. Python runner（T100 で追加）
T100 の Python runner で UI 構造を自動チェックする（jq 不要）。
構造・存在・commit pin・HyperParameters セクション分割のみを検証し、
グラフの内容は本チェックリストで目視する。

```bash
python tools/tests/rehearsal_verify_clearml_ui.py --usecase-id <usecase_id>
```

任意オプション:
```bash
python tools/tests/rehearsal_verify_clearml_ui.py --usecase-id <usecase_id> --project-root LOCAL
python tools/tests/rehearsal_verify_clearml_ui.py --usecase-id <usecase_id> --json
```

- pipeline 不在 / train 不在は終了コード 1
- version_num pin や HyperParameters が General 一括のままの場合は WARNING
