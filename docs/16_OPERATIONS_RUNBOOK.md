# 16 Operations Runbook

このファイルは、operator が日常運用で使う最短手順をまとめた runbook です。  
想定する流れは「seed pipeline を同期し、`.pipelines/<profile>` で確認し、`NEW RUN` で実行し、結果と推奨モデルを確認する」です。

## 1. 前提条件

- `CLEARML_CONFIG_FILE` が有効
- ClearML UI に接続できる
- controller / child / heavy-child の役割に対応する queue が存在する
- agent は `tools/clearml_agent/compose.yaml` を正本として起動している
- agent の `/root/.clearml` は Docker named volume を使う
- Windows bind mount は task repository / venv / uv cache が `p9_client_rpc` 待ちになりやすいので避ける

## 2. seed pipeline を同期する

```bash
python tools/clearml_templates/manage_templates.py --apply --project-root LOCAL
python tools/clearml_templates/manage_templates.py --validate --project-root LOCAL
```

確認ポイント:

- seed pipeline が `LOCAL/TabularAnalysis/.pipelines/<profile>` にある
- `pipeline`
- `train_model_full`
- `train_ensemble_full`

## 3. UI から実行する

### 推奨

1. ClearML UI で対象 profile の seed pipeline project を開く
2. `LOCAL/TabularAnalysis/.pipelines/<profile>` の seed card を確認する
3. 必要な card を開いて `NEW RUN` を選ぶ
4. `Configuration > OperatorInputs` を先に確認し、`run.usecase_id` と `data.raw_dataset_id` の seed 既定値を把握する
5. seed card の `data.raw_dataset_id` が placeholder `REPLACE_WITH_EXISTING_RAW_DATASET_ID` でも正常なので、実編集は `Hyperparameters` 側で既存 raw dataset id へ置き換える
6. `run.usecase_id` を明示しない場合は seed 既定値 `TabularAnalysis` のままでもよく、その場合は runtime が `run.usecase_id_policy` に従って actual run 用の一意な値へ自動採番する
7. 実行する

### seed profile の使い分け

- `pipeline`
  - preprocess + single-model train + leaderboard
- `train_model_full`
  - preprocess + single-model train
- `train_ensemble_full`
  - preprocess + single-model train + 3 ensemble + leaderboard

注意:

- seed pipeline の DAG は profile ごとに固定です
- 標準 pipeline は `data.raw_dataset_id` 指定前提で、`dataset_register` は準備系導線です
- UI から安全に編集する対象は `run.usecase_id`, `data.raw_dataset_id`, `pipeline.selection.enabled_preprocess_variants`, `pipeline.selection.enabled_model_variants` に絞ります
- `run.usecase_id` は毎回固有にするのが推奨です。未編集で seed 既定値 `TabularAnalysis` のまま起動した場合も、actual run では自動採番されるため古い task と混ざりにくくなります
- `Configuration > OperatorInputs` は確認用 mirror です。実際に値を書き換える場所は `Hyperparameters` です
- `train_ensemble_full` だけは追加で `ensemble.selection.enabled_methods`, `ensemble.top_k` を編集対象にします
- `pipeline.model_set` や `pipeline.grid.model_variants` を UI で変えて custom graph を作る運用は行いません
- custom な task 組み合わせを試すときは developer 向けの CLI / config 変更で扱います

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
  data.raw_dataset_id=<RAW_DATASET_ID> \
  pipeline.selection.enabled_preprocess_variants=[stdscaler_ohe] \
  pipeline.selection.enabled_model_variants=[ridge,lgbm,xgboost]
```

### ensemble ありの full run

```bash
python -m tabular_analysis.cli task=pipeline \
  run.clearml.enabled=true \
  run.clearml.execution=pipeline_controller \
  run.clearml.project_root=LOCAL \
  +pipeline.profile=train_ensemble_full \
  data.raw_dataset_id=<RAW_DATASET_ID> \
  pipeline.selection.enabled_preprocess_variants=[stdscaler_ohe] \
  pipeline.model_set=regression_all \
  ensemble.selection.enabled_methods=[mean_topk,weighted,stacking]
```

## 5. 実行後に見る場所

### ClearML UI

- Pipeline seed
  - `LOCAL/TabularAnalysis/.pipelines/<profile>`
- Pipeline run
  - `LOCAL/TabularAnalysis/Pipelines/Runs/<usecase_id>`
- child tasks
  - `LOCAL/TabularAnalysis/Runs/<usecase_id>/01_Datasets`
  - `LOCAL/TabularAnalysis/Runs/<usecase_id>/02_Preprocess`
  - `LOCAL/TabularAnalysis/Runs/<usecase_id>/03_TrainModels`
  - `LOCAL/TabularAnalysis/Runs/<usecase_id>/04_Ensembles`
  - `LOCAL/TabularAnalysis/Runs/<usecase_id>/05_Infer`
  - `LOCAL/TabularAnalysis/Runs/<usecase_id>/05_Infer_Children`
  - `LOCAL/TabularAnalysis/Runs/<usecase_id>/99_Leaderboard`

### ローカル出力

- `work/rehearsal/out/...`
- task の `run.output_dir`

### 主な成果物

- `pipeline_run.json`
- `run_summary.json`
- `report.md`
- `report.json`
- `report_links.json`

## 6. 推奨モデルの確認

見るファイル:

- `05_leaderboard/recommendation.json`
- pipeline の `report.json`
- `99_Leaderboard` task の `PLOTS -> leaderboard/table`

見る項目:

主表示:

- `recommended_infer_key`
- `recommended_infer_value`
- `recommended_ref_kind`

補助表示:

- `recommended_registry_model_id`
- `recommended_train_task_id`
- `recommended_model_id`
- `primary_metric`
- `best_score`

`99_Leaderboard` の `PLOTS -> leaderboard/table` では、各行に次の列が出る。

- `ref_kind`
  - `model_id` なら `infer.model_id` を使う
  - `train_task_id` なら `infer.train_task_id` を使う
- `infer_key`
  - UI の infer task で設定する Hyperparameter key
- `infer_value`
  - そのままコピーして使う id

最短手順は、`rank=1` の行か `recommendation.json` の `recommended_infer_key` / `recommended_infer_value` をそのまま infer task の `Hyperparameters` に入れること。

迷ったときの判断基準:

- `recommended_infer_key` / `recommended_infer_value`: 実際に入力する正本
- `recommended_registry_model_id`: 昇格済みモデルとして使うか確認するときに見る
- `recommended_train_task_id`: 実験 run を再現したいときに見る
- `recommended_model_id`: 互換・診断用で、通常操作の主入力にはしない

## 7. 推論する

registry model id を使う例:

```bash
python -m tabular_analysis.cli task=infer \
  run.clearml.enabled=true \
  run.clearml.execution=logging \
  infer.mode=single \
  infer.model_id=<CLEARML_REGISTRY_MODEL_ID> \
  infer.input_json='{"num1":1.0,"num2":2.0,"cat":"a"}'
```

## 8. 困ったときの確認順

1. `manage_templates --validate`
2. queue に worker がいるか
3. controller が controller role の queue に載っているか
4. `catboost/xgboost` が heavy-child role の queue に振り分けられているか
5. child task の tags が runtime 用に更新されているか

## 8.5 rehearsal helper の既定動作

`python tools/rehearsal/run_pipeline_v2.py --execution agent ...` は、既定で remote controller の完了まで待機します。

- enqueue 直後の launch stub では終わりません
- controller 完了後に `99_pipeline/pipeline_run.json`, `run_summary.json`, `report.json`, `report_links.json`, `report.md` を local output に同期します
- 待たずに enqueue のみ行いたい場合は `--no-wait` を使います

## 9. 関連ドキュメント

- [10_OPERATION_MODES.md](10_OPERATION_MODES.md)
- [67_REHEARSAL_COMMANDS.md](67_REHEARSAL_COMMANDS.md)
- [68_CLEARML_AGENT_TROUBLESHOOTING.md](68_CLEARML_AGENT_TROUBLESHOOTING.md)
- [69_CLEARML_TROUBLESHOOTING.md](69_CLEARML_TROUBLESHOOTING.md)
- [81_CLEARML_TEMPLATE_POLICY.md](81_CLEARML_TEMPLATE_POLICY.md)
- [82_CLEARML_PROJECT_LAYOUT.md](82_CLEARML_PROJECT_LAYOUT.md)

