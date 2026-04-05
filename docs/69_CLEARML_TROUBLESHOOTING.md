# 69 ClearML Troubleshooting

## まず確認する順番

1. seed / template を apply / validate したか
2. queue に healthy worker がいるか
3. task script の repo / branch / entrypoint は正しいか
4. project / tags / metadata は想定どおりか
5. optional dependency は足りているか

## seed / template refresh

```bash
python tools/clearml_templates/manage_templates.py --apply --project-root LOCAL
python tools/clearml_templates/manage_templates.py --validate --project-root LOCAL
```

## よくある症状

### task が queued のまま

- queue に worker がいない
- queue 名が違う
- `controller/default/heavy-model` の役割がずれている

### pipeline が Pipelines タブに出ない

- `LOCAL/TabularAnalysis/.pipelines/<profile>` に seed pipeline card がない
- seed pipeline の `process:pipeline` / `task_kind:seed` / `pipeline_profile:<name>` が崩れている
- seed pipeline が `TaskTypes.controller` でない

### child task の tags が汚れている

- runtime tag rebuild が効いていない
- `template:true` や stale `usecase` が残っている

### model bootstrap が重すぎる

- optional extra が model 単位になっていない
- `lightgbm/xgboost/catboost` を不要に同時 install している

## script 確認

```bash
python - <<'PY'
from clearml import Task
task = Task.get_task(task_id="<TASK_ID>")
print(task.get_script())
PY
```

見るべき点:

- `repository`
- `branch` or `version_num`
- `entry_point=tools/clearml_entrypoint.py`


