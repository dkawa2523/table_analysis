# ClearML: Pipeline Controller 実装契約 v2

## 問題
`pipeline` を実行しても「pipeline という名前の Task だけ」になり、各工程の Task が作成されない。

## 目的
- ClearML の **PipelineController** を使って、preprocess / train_model / train_ensemble / leaderboard / infer を **別Taskとして生成・実行**する。
- 複数モデル学習（grid）を controller で束ね、UI上で追跡しやすくする（ensemble は有効時のみ生成）。
- **設計の冗長化を防ぐ**: local pipeline と controller pipeline の“仕様”を共通化し、分岐だけで動くようにする。

## 実行モード
- `run.clearml.execution=pipeline_controller` を追加し、このモードのとき:
  - pipeline task は controller の orchestrator としてのみ動く
  - 子タスク群が ClearML 上に作成され、queue に投入される
- `execution=logging` はローカル逐次実行（driver=local_sequential）で各工程が別タスクとして記録される

## 入力前提
- pipeline は `data.raw_dataset_id` を必須入力として受け取り、dataset_register は **別タスク**で実行する。

## 子タスク作成方針（試験段階）
- 原則: pipeline/子タスクとも **template task を clone** してパラメータを上書き（UI運用に近い）
- template 探索は tags と project を用いて安定化する（詳細は `docs/81_CLEARML_TEMPLATE_POLICY.md`）
  - 必須: `template:true` / `process:<name>` / `template_set:<id>` / `solution:tabular-analysis`
  - 優先: `usecase:<template_usecase_id>` / `schema:<schema_version>`
  - 複数一致は **明示的にエラー**（事故防止）
- template が無い場合は明確にエラー（「テンプレ作成ツールを先に実行せよ」を表示）
- controller は commit pin を使わず、template の branch/entry_point を優先する

## grid 実行
- preprocess_variants を展開し、各 preprocess の出力 `processed_dataset_id` に依存する train を作成
- model_variants を展開し、train tasks を並列に
- ensemble.enabled=true の場合、preprocess ごとに train_ensemble を 1 つ生成する
- tags は `docs/66_NAMING_TAGGING_POLICY.md` に従う（`grid:<grid_run_id>` / `grid_cell:<preprocess>__<model>` など）

## Orchestrator 出力
- pipeline task は `pipeline_run.json` を artifact に残し、子タスクIDの参照を保持する
- 追加 artifacts（plan/report/run_summary）は `docs/03_CLEARML_UI_CONTRACT.md` / `docs/60_PIPELINE_TRAIN_CONTRACT.md` を単一の正とする
- Scalars:
  - `pipeline/num_models`
  - `pipeline/num_succeeded`
  - `pipeline/num_failed`

## plan + driver 分離
- plan は ClearML 非依存の純粋関数で生成し、driver が **同一 plan** を消費する
- local_sequential / pipeline_controller で project/tags/hparams の見え方を揃える
- parent task_id は driver 側で注入し、plan 自体は再利用可能に保つ

## Hyperparameters / Configuration
- 子タスクは `clearml/hparams.py` を使い、UIの HyperParameters に「最低限の再現情報」を載せる（詳細は docs/53）。
