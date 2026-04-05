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

### `NEW RUN` 直後に placeholder で止まる

- seed card の `data.raw_dataset_id` は placeholder `REPLACE_WITH_EXISTING_RAW_DATASET_ID` が正常
- 実編集は `Hyperparameters` 側で行う必要がある
- actual run では placeholder のまま開始すると fail-fast する

### Hyperparameters に `%2E` が見える

- 古い historical run では ClearML が保存した cloned payload の都合で `%2E` を含む key が残ることがある
- current seed と新規 `NEW RUN` は nested section を正本にし、encoded key を section sync で置き換える
- current task の source-of-truth は引き続き `Args/*` だが、operator が見るべき section UI では `%2E` を増やさない
- 古い run の表示ノイズは current runtime failure とは限らない

確認ポイント:

- seed / run task の `Hyperparameters` で
  - `inputs -> run -> usecase_id`
  - `dataset -> data -> raw_dataset_id`
  - `selection -> pipeline -> selection -> enabled_model_variants`
  のように nested 表示されているか
- `Configuration > OperatorInputs` が current values を mirror しているか
- `%2E` が見えるのが old historical run だけか、それとも current seed / current `NEW RUN` にも出ているか

current seed / current `NEW RUN` にも `%2E` が出る場合は、次を疑います。

- `manage_templates --apply` 前の古い seed がまだ見えている
- current task 正規化より後で flat dotted payload を再接続している
- agent が古い commit の `tools/clearml_entrypoint.py` を実行している

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


