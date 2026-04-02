# 推論（infer） batch / optimize 契約 v1（controllerは使わない）

## ゴール
- infer.mode=single : 単独タスク
- infer.mode=batch : サマリータスク + 条件ごとの child 推論タスク（single相当）
- infer.mode=optimize : サマリータスク + trialごとの child 推論タスク（single相当、Optunaで探索）

## 制約
- 推論は PipelineController を使わない（ユーザー要求）。
- ただし ClearML Task を clone/enqueue して child task を作ることは許容（実装簡潔のため）。

## UI 出力
- child task: 各条件の入力→出力テーブルを PLOTS に出す（Plotly Table）
- summary task:
  - batch: 全条件のテーブル + 予測分布など
  - optimize: Optuna可視化（history/parallel coords/importance/response surface） + 上位条件テーブル（input_output_table）
- project 配置:
  - summary: `<project_root>/TabularAnalysis/<usecase_id>/05_Infer`
  - child: `<project_root>/TabularAnalysis/<usecase_id>/05_Infer_Children`

## 入力と上限
- batch summary は `infer.batch.inputs_path`（csv/parquet）または `infer.batch.inputs_json`（条件リスト）を読む
- `infer.batch.max_children` で child タスクの上限 N を指定（超過分は先頭 N 件のみ）
- `infer.batch.wait_timeout_sec`/`infer.batch.poll_interval_sec` で待機時間を制御（timeout は集約に `timeout` として残る）
  - optimize は `infer.optimize.wait_timeout_sec`/`infer.optimize.poll_interval_sec` を使う（未設定時は batch 値を使用）

## 注意事項
- `run.clearml.queue_name`（または `exec_policy.queues.infer`）が未設定だと enqueue できない
- 入力件数が多い場合は `infer.batch.max_children` でタスク爆発を防ぐ
- infer.mode=optimize は Optuna が必須（未導入の場合は明示エラー）
- optimize の child task は **親の infer.optimize.\*** を引き継がない（override で明示的にクリアする）

## ClearML テンプレ／override 運用
- clone は常に `template:true` の最新テンプレから行う（旧テンプレは `template:deprecated`）
- JSON override（`infer.input_json` / `infer.batch.inputs_json` / `infer.optimize.search_space`）は
  `tools/clearml_entrypoint.py` が自動クォートして Hydra パース失敗を防ぐ
- child task の `run.clearml.task_name` は override で付与してよい
  （`conf/run/base.yaml` に `run.clearml.task_name` を定義済み）
- optimize の child は `trial:optimize` + `parent:<task_id>` タグを付与して UI 検索しやすくする

## optimize 設定
- `infer.optimize.n_trials`: 試行回数
- `infer.optimize.direction`: maximize/minimize
- `infer.optimize.sampler`: tpe/random/cmaes
- `infer.optimize.objective.key`: 目的関数の参照キー（例: prediction, pred_proba_1）
- `infer.optimize.search_space`: 探索空間（連続/離散）
  - list 形式例:
    - `{name: feature_a, type: float, low: 0.0, high: 1.0}`
    - `{name: feature_b, type: int, low: 0, high: 10, step: 1}`
    - `{name: feature_c, type: categorical, choices: [a, b, c]}`
  - mapping 形式も許容（key が name）

## HyperParameters
- summary: Inputs/Optimize などをカテゴリ別に
- child: Inputs/Model/Dataset を最小で
  - Inputs: infer.mode, input.source, input.path/json, schema_policy
  - Model: model_id, model_abbr
  - Dataset: train_task_id, raw/processed dataset_id, preprocess_variant, split_hash, recipe_hash
