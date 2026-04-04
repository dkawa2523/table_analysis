# 16 Operations Runbook

このファイルは、operator が日常運用で使う最短手順をまとめた runbook です。  
想定する流れは「template を同期し、visible pipeline template を確認し、実行し、結果と推奨モデルを確認する」です。

## 1. 前提条件

- `CLEARML_CONFIG_FILE` が有効
- ClearML UI に接続できる
- `services`、`default`、`heavy-model` queue が存在する
- agent は `tools/clearml_agent/compose.yaml` を正本として起動している
- agent の `/root/.clearml` は Docker named volume を使う
  - Windows bind mount は task repository / venv / uv cache が `p9_client_rpc` 待ちになりやすいので避ける

## 2. template を同期する

```bash
python tools/clearml_templates/manage_templates.py --apply --project-root LOCAL
python tools/clearml_templates/manage_templates.py --validate --project-root LOCAL
```

確認ポイント:

- visible pipeline template が `LOCAL/TabularAnalysis/Pipelines` にある
- `pipeline`
- `train_model_full`
- `train_ensemble_full`

## 3. UI から実行する

### 推奨

1. ClearML の `Pipelines` タブを開く
2. `LOCAL/TabularAnalysis/Pipelines` の template を確認する
3. 必要な template を clone する
4. dataset id や usecase などを編集する
5. 実行する

### template の使い分け

- `pipeline`
  - preprocess + single-model train + leaderboard
- `train_model_full`
  - preprocess + single-model train
- `train_ensemble_full`
  - preprocess + single-model train + 3 ensemble + leaderboard

## 4. CLI から実行する

### raw dataset を登録

```bash
python -m tabular_analysis.cli task=dataset_register \
  run.clearml.enabled=true \
  run.clearml.execution=logging \
  run.clearml.project_root=LOCAL \
  data.dataset_path=/path/to/data.csv \
  data.target_column=target
```

### pipeline controller を起動

```bash
python -m tabular_analysis.cli task=pipeline \
  run.clearml.enabled=true \
  run.clearml.execution=pipeline_controller \
  run.clearml.project_root=LOCAL \
  run.clearml.queue_name=services \
  data.raw_dataset_id=<RAW_DATASET_ID> \
  +pipeline.preprocess_variant=stdscaler_ohe \
  +pipeline.model_set=regression_all
```

### ensemble ありの full run

```bash
python -m tabular_analysis.cli task=pipeline \
  run.clearml.enabled=true \
  run.clearml.execution=pipeline_controller \
  run.clearml.project_root=LOCAL \
  run.clearml.queue_name=services \
  data.raw_dataset_id=<RAW_DATASET_ID> \
  +pipeline.preprocess_variant=stdscaler_ohe \
  +pipeline.model_set=regression_all \
  pipeline.run_train_ensemble=true \
  ensemble.enabled=true \
  ensemble.methods=[mean_topk,weighted,stacking]
```

## 5. 実行後に見る場所

### ClearML UI

- Pipeline template / run
  - `LOCAL/TabularAnalysis/Pipelines`
- child tasks
  - `LOCAL/TabularAnalysis/<usecase_id>/...`

### ローカル出力

- `work/rehearsal/out/...`
- task の `run.output_dir`

### 主な成果物

- `pipeline_run.json`
- `report.md`
- `report.json`
- `report_links.json`
- `run_summary.json`

## 6. 推奨モデルの確認

見るファイル:

- `05_leaderboard/recommendation.json`
- pipeline の `report.json`

見る項目:

- `recommended_train_task_id`
- `recommended_model_id`
- `infer_model_id`
- `primary_metric`
- `best_score`

## 7. 推論する

registry model id を使う例:

```bash
python -m tabular_analysis.cli task=infer \
  run.clearml.enabled=true \
  run.clearml.execution=logging \
  infer.mode=single \
  infer.model_id=<CLEARML_REGISTRY_MODEL_ID> \
  infer.single.input_json='{"num1":1.0,"num2":2.0,"cat":"a"}'
```

## 8. 困ったときの確認順

1. `manage_templates --validate`
2. queue に worker がいるか
3. controller が `services` に載っているか
4. `catboost/xgboost` が `heavy-model` に振り分けられているか
5. child task の tags が runtime 用に更新されているか

## 9. 関連ドキュメント

- [10_OPERATION_MODES.md](10_OPERATION_MODES.md)
- [67_REHEARSAL_COMMANDS.md](67_REHEARSAL_COMMANDS.md)
- [68_CLEARML_AGENT_TROUBLESHOOTING.md](68_CLEARML_AGENT_TROUBLESHOOTING.md)
- [69_CLEARML_TROUBLESHOOTING.md](69_CLEARML_TROUBLESHOOTING.md)
- [81_CLEARML_TEMPLATE_POLICY.md](81_CLEARML_TEMPLATE_POLICY.md)
- [82_CLEARML_PROJECT_LAYOUT.md](82_CLEARML_PROJECT_LAYOUT.md)

