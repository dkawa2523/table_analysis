# 55 ClearML UI Checklist

このチェックリストは、rehearsal や pipeline 実行後に ClearML UI 上で何を確認すべきかをまとめたものです。  
operator と reviewer の両方が使えるよう、task ごとに確認ポイントを分けています。

## 1. seed pipeline task

- [ ] `LOCAL/TabularAnalysis/.pipelines/<profile>` に seed pipeline card がある
- [ ] `pipeline`
- [ ] `train_model_full`
- [ ] `train_ensemble_full`
- [ ] type が `controller`
- [ ] task tags に `process:pipeline`, `task_kind:seed`, `pipeline_profile:<name>` がある
- [ ] project に `pipeline` system tag が付いている

## 2. dataset_register

- [ ] project が `<project_root>/TabularAnalysis/Runs/<usecase_id>/01_Datasets`
- [ ] tags に `usecase:<id>`, `process:dataset_register`, `schema:v1`
- [ ] artifacts に `config_resolved.yaml`, `out.json`, `manifest.json`, `schema.json`, `preview.csv`
- [ ] HyperParameters に dataset / clearml の section がある

## 3. preprocess

- [ ] project が `<project_root>/TabularAnalysis/Runs/<usecase_id>/02_Preprocess`
- [ ] tags に `process:preprocess`, `preprocess:<variant>`
- [ ] artifacts に `preprocess_bundle.*`, `recipe.json`, `split.json`, `summary.md`
- [ ] processed dataset が作成されている
- [ ] `split_hash`, `recipe_hash` が property か out.json に残っている

## 4. train_model

- [ ] project が `<project_root>/TabularAnalysis/Runs/<usecase_id>/03_TrainModels`
- [ ] tags に `process:train_model`, `model:<variant>`, `grid:<id>`
- [ ] heavy model が必要なら queue が期待どおり
- [ ] artifacts に `model_bundle.*`, `metrics.json`, `preds_valid.parquet`
- [ ] scalars に primary metric がある
- [ ] plots に residual / feature importance などがある

## 5. train_ensemble

- [ ] project が `<project_root>/TabularAnalysis/Runs/<usecase_id>/04_Ensembles`
- [ ] tags に `process:train_ensemble`, `ensemble:<method>`
- [ ] artifacts に `ensemble_spec.json`, `model_bundle.joblib`, `metrics.json`

## 6. leaderboard

- [ ] project が pipeline 配下または usecase 配下の expected location にある
- [ ] artifacts に `leaderboard.csv`, `recommendation.json`, `summary.md`
- [ ] recommendation に `recommended_train_task_id`, `recommended_model_id`, `infer_model_id` がある
- [ ] comparability 情報が report に残っている

## 7. infer

- [ ] project が `<project_root>/TabularAnalysis/Runs/<usecase_id>/05_Infer`
- [ ] batch child がある場合は `05_Infer_Children`
- [ ] `infer.model_id` か `infer.train_task_id` のどちらで動いたか読み取れる
- [ ] predictions artifact がある

## 8. pipeline controller

- [ ] run controller が `LOCAL/TabularAnalysis/Pipelines/Runs/<usecase_id>` に見える
- [ ] type が `controller`
- [ ] `task_kind:run` がある
- [ ] `template:true` が残っていない
- [ ] `usecase:<actual>` が正しく付いている
- [ ] child task に stale usecase や template tag が残っていない

## 9. HyperParameters

- [ ] `inputs`
- [ ] `dataset`
- [ ] `preprocess`
- [ ] `model`
- [ ] `eval`
- [ ] `pipeline`
- [ ] `clearml`

必要な section が空でもよいが、`General` へ全部落ちていないことを確認する。

## 10. report artifacts

- [ ] `pipeline_run.json`
- [ ] `report.md`
- [ ] `report.json`
- [ ] `report_links.json`
- [ ] `run_summary.json`

## 11. queue / worker 観点

- [ ] controller は `controller`
- [ ] light child は `default`
- [ ] `catboost/xgboost` は `heavy-model`

## 12. 最低限の operator 完了条件

- [ ] `.pipelines/<profile>` project に seed pipeline card が見える
- [ ] seed pipeline card から `NEW RUN` できる
- [ ] seed card の `Configuration > OperatorInputs` に placeholder `REPLACE_WITH_EXISTING_RAW_DATASET_ID` が見えても仕様どおりである
- [ ] cloned run では `Hyperparameters` 側の `data.raw_dataset_id` が実値へ更新されている
- [ ] `Configuration > OperatorInputs` は確認用 mirror、実編集は `Hyperparameters` という役割で理解できる
- [ ] `run.usecase_id` を seed 既定値 `TabularAnalysis` のまま起動した場合でも、actual run では一意な `<usecase_id>` に切り替わる
- [ ] run controller が `LOCAL/TabularAnalysis/Pipelines/Runs/<usecase_id>` に残る
- [ ] 推奨モデルと report が取れる
- [ ] child task を UI から辿れる
- [ ] 新規 run では `%2E` を含む legacy key が主表示に出ていない

