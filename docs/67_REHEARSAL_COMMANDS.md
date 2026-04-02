# 67 Rehearsal Commands

## Canonical rehearsal runner
```bash
python tools/rehearsal/run_pipeline_v2.py --execution local \
  --task-type regression --preprocess stdscaler_ohe --models ridge,elasticnet

python tools/rehearsal/run_pipeline_v2.py --execution logging \
  --task-type regression --preprocess stdscaler_ohe --models ridge,elasticnet \
  --project-root LOCAL

python tools/rehearsal/run_pipeline_v2.py --execution agent --queue-name default \
  --task-type regression --preprocess stdscaler_ohe --models ridge,elasticnet \
  --project-root LOCAL
```

## ClearML UI verification
```bash
python tools/tests/rehearsal_verify_clearml_ui.py --usecase-id <USECASE_ID> --project-root LOCAL
```

## Manual task commands

### dataset_register
```bash
python -m tabular_analysis.cli task=dataset_register \
  run.clearml.enabled=true run.clearml.execution=logging \
  data.dataset_path=/tmp/ta_rehearsal_data/toy_reg.csv \
  data.target_column=target
```

### pipeline
```bash
python -m tabular_analysis.cli task=pipeline \
  run.clearml.enabled=true run.clearml.execution=logging \
  data.raw_dataset_id=<RAW_DATASET_ID> \
  pipeline.preprocess_variant=stdscaler_ohe \
  +pipeline.model_set=regression_all
```

### pipeline controller
```bash
python -m tabular_analysis.cli task=pipeline \
  run.clearml.enabled=true run.clearml.execution=pipeline_controller \
  run.clearml.queue_name=default \
  data.raw_dataset_id=<RAW_DATASET_ID> \
  pipeline.preprocess_variant=stdscaler_ohe \
  +pipeline.model_set=regression_all
```

### infer
```bash
python -m tabular_analysis.cli task=infer infer.mode=single \
  infer.model_id=<MODEL_ID> infer.input_path=/tmp/infer.csv

python -m tabular_analysis.cli task=infer infer.mode=batch \
  infer.model_id=<MODEL_ID> infer.batch.inputs_path=/tmp/batch_inputs.csv
```

## Template refresh
```bash
python tools/clearml_templates/manage_templates.py --plan --project-root LOCAL
python tools/clearml_templates/manage_templates.py --apply --project-root LOCAL
python tools/clearml_templates/manage_templates.py --validate --project-root LOCAL
```
