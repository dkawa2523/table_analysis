# 67 Rehearsal Commands

このファイルは「まず動かすためのコマンド集」です。  
実運用前の smoke、UI 契約確認、template refresh、推論確認を短時間で回すときに使います。

## 1. 最短のローカル回帰 pipeline

```bash
python tools/rehearsal/run_pipeline_v2.py \
  --execution local \
  --task-type regression \
  --preprocess stdscaler_ohe \
  --models ridge,elasticnet
```

## 2. ClearML に記録しながら回す

```bash
python tools/rehearsal/run_pipeline_v2.py \
  --execution logging \
  --task-type regression \
  --preprocess stdscaler_ohe \
  --models ridge,elasticnet \
  --project-root LOCAL
```

## 3. visible pipeline template を使う controller 実行

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

注意:

- `pipeline_controller` の正規 queue は `controller`
- child task は `default` / `heavy-model` に流れる

## 4. template の同期

```bash
python tools/clearml_templates/manage_templates.py --plan --project-root LOCAL
python tools/clearml_templates/manage_templates.py --apply --project-root LOCAL
python tools/clearml_templates/manage_templates.py --validate --project-root LOCAL
```

repo / branch を明示する例:

```bash
python tools/clearml_templates/manage_templates.py \
  --apply --project-root LOCAL \
  --repo https://github.com/dkawa2523/table_analysis \
  --branch main
```

## 5. 手動で task を回す

## dataset_register

```bash
python -m tabular_analysis.cli task=dataset_register \
  run.clearml.enabled=true \
  run.clearml.execution=logging \
  run.clearml.project_root=LOCAL \
  data.dataset_path=/path/to/toy_reg.csv \
  data.target_column=target
```

## preprocess

```bash
python -m tabular_analysis.cli task=preprocess \
  run.clearml.enabled=false \
  data.dataset_path=/path/to/toy_reg.csv \
  data.target_column=target \
  preprocess.variant=stdscaler_ohe
```

## train_model

```bash
python -m tabular_analysis.cli task=train_model \
  run.clearml.enabled=false \
  train.inputs.preprocess_run_dir=/path/to/preprocess_run \
  group/model=ridge
```

## pipeline

```bash
python -m tabular_analysis.cli task=pipeline \
  run.clearml.enabled=true \
  run.clearml.execution=logging \
  run.clearml.project_root=LOCAL \
  data.raw_dataset_id=<RAW_DATASET_ID> \
  +pipeline.preprocess_variant=stdscaler_ohe \
  +pipeline.model_set=regression_all
```

## pipeline controller

```bash
python -m tabular_analysis.cli task=pipeline \
  run.clearml.enabled=true \
  run.clearml.execution=pipeline_controller \
  run.clearml.project_root=LOCAL \
  data.raw_dataset_id=<RAW_DATASET_ID> \
  +pipeline.preprocess_variant=stdscaler_ohe \
  +pipeline.model_set=regression_all
```

## full ensemble pipeline

```bash
python -m tabular_analysis.cli task=pipeline \
  run.clearml.enabled=true \
  run.clearml.execution=pipeline_controller \
  run.clearml.project_root=LOCAL \
  data.raw_dataset_id=<RAW_DATASET_ID> \
  +pipeline.preprocess_variant=stdscaler_ohe \
  +pipeline.model_set=regression_all \
  pipeline.run_train_ensemble=true \
  ensemble.enabled=true \
  ensemble.methods=[mean_topk,weighted,stacking]
```

## infer

single:

```bash
python -m tabular_analysis.cli task=infer \
  infer.mode=single \
  infer.model_id=<MODEL_ID> \
  infer.single.input_json='{"num1":1.2,"num2":3.4,"cat":"a"}'
```

batch:

```bash
python -m tabular_analysis.cli task=infer \
  infer.mode=batch \
  infer.model_id=<MODEL_ID> \
  infer.batch.inputs_path=/tmp/batch_inputs.csv
```

## 6. UI 契約の確認

```bash
python tools/tests/rehearsal_verify_clearml_ui.py \
  --usecase-id <USECASE_ID> \
  --project-root LOCAL
```

## 7. agent 起動

```bash
docker compose -f tools/clearml_agent/compose.yaml up -d --build
```

追加の host mount 変数は不要です。  
canonical compose は Docker named volume で `/root/.clearml` と `UV_CACHE_DIR=/root/.clearml/uv-cache` を管理します。

## 8. 出力を見る

rehearsal runner の既定出力:

```text
work/rehearsal/out/<execution>/<usecase_id>/
```

よく見るファイル:

- `rehearsal_summary.json`
- `99_pipeline/report.json`
- `99_pipeline/report_links.json`
- `05_leaderboard/recommendation.json`

## 9. よく使うテスト

```bash
python tools/tests/verify_all.py --quick
python tools/tests/check_docs_paths.py --repo .
python tools/tests/test_template_specs.py
python tools/tests/test_clearml_runtime_contracts.py
```

