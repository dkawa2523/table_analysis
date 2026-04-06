# Rehearsal Guide

このドキュメントは、UI と local smoke の両方で ClearML pipeline を安全にリハーサルするための要点をまとめる。

## 学習 pipeline の UI リハーサル

1. `Pipelines` タブで seed card を開く
2. `NEW RUN`
3. `Configuration > OperatorInputs` で確認
4. `Hyperparameters` で `data.raw_dataset_id` を実値に変更
5. 実行

## leaderboard から infer へ進む

1. `99_Leaderboard` を開く
2. `PLOTS -> leaderboard/table` で `infer_key` と `infer_value` を確認
3. または `recommendation.json` から `infer_model_id` / `infer_train_task_id` を確認
4. `Templates/Steps/05_Infer` の `infer` task を clone
5. `Hyperparameters` に `infer_key=infer_value` を設定して実行

## local smoke

- `python tools/tests/smoke_local.py --repo . --until pipeline`
- `python tools/tests/test_pipeline_report.py`
- `python tools/tests/test_leaderboard_ui_contract.py`

## よくある確認点

- seed は `.pipelines/<profile>` にあるか
- actual run は `Pipelines/Runs/<usecase_id>` に作られるか
- child task は `Runs/<usecase_id>/<group>` に分かれるか
- fresh task に `%2E` key が残っていないか
