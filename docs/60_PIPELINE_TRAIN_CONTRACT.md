# 学習パイプライン（回帰） 契約 v1（update-3_clearml）

## 目的
- pipeline は **ClearML Dataset 登録から始めない**。ユーザーは事前に `dataset_register` を実行し、`raw_dataset_id` を得る。
- pipeline は `raw_dataset_id`（= ClearML Dataset ID）を入力として、以下を一括実行する:
  1) preprocess（選択した前処理 variant）
  2) train_model（回帰モデルを複数。基本は「回帰ほぼ全て」= model_set）
  3) train_ensemble（`ensemble.enabled=true` の場合のみ）
  4) leaderboard（単体 + アンサンブルを比較・評価・推薦）
  5) infer（`pipeline.run_infer=true` の場合のみ）
- ClearML 上では **個別タスクとして実行され、追跡・比較が可能**であること。

## 入力
- 必須: `data.raw_dataset_id`
- 任意: `data.dataset_path`（`raw_dataset_id` が `local:` の場合のみ必要）
- 任意: `pipeline.preprocess_variant`
- 任意: `pipeline.model_set`（例: `regression_all`）または `pipeline.model_variants`（明示リスト）
- 任意: `eval.metrics`（標準: R2, MSE, RMSE, MAE）

### model_set の解決
- `pipeline.model_set` は `conf/pipeline/model_sets/<name>.yaml` を参照し、`pipeline.model_variants` に展開される。
- `auto: true` の場合は registry の `list_model_variants(task_type=...)` で自動列挙する（`model_variant.class_path` の regression/classification 定義 or `model_variant.task_type` を参照）。
- 固定リスト運用にしたい場合は `variants` を明示し、新しい回帰モデルを追加したら `conf/pipeline/model_sets/regression_all.yaml` を更新する。

## 出力
- preprocess: `processed_dataset_id`（ClearML Dataset）
- train_model: `model_id`（ClearML Model Registry 参照） + 指標（Scalars/Plots）
- leaderboard: `leaderboard.csv` + `recommendation.json`（推奨モデル）
- pipeline（親タスク）:
  - `pipeline_run.json`, `plan.json`
  - `report.md`, `report.json`, `report_links.json`
  - `run_summary.json`（partial failure の集計/推薦/リンク）

## 運用上の前提
- dataset_register はローカル実行で良い（試験段階）。社内サーバ移行時も同じ。
- pipeline は `data.raw_dataset_id` 入力を前提とし、`data.dataset_path` は dataset_register のみに使う。
- pipeline は `run.clearml.execution=local/logging` でローカル逐次、`pipeline_controller` で agent 実行する。
- `run.clearml.execution=local` でも `run.clearml.enabled=true` の場合は子タスクを logging として記録する。
- ローカルで一括実行したい場合は、別途 `local_orchestrator` を使う（docs/67）。
- agent 実行は template clone 前提で local/logging と見え方を揃える（`docs/10_OPERATION_MODES.md` / `docs/81_CLEARML_TEMPLATE_POLICY.md`）。

## タスク名 / タグ / Properties
- 命名/タグ/Properties は `docs/66_NAMING_TAGGING_POLICY.md` を単一の正とする。
- pipeline 実行時は `run.grid_run_id` が各タスクに伝播し、`grid:<grid_run_id>` が付与される。

## pipeline v2 スキーマ（T089: 先行定義）
- `pipeline.profile` / `pipeline.groups.*` / `pipeline.fail_policy.*` / `pipeline.limits.*` / `pipeline.parallelism.*` を追加で定義した。

## pipeline v2 設計（profile / groups / base+差分）
レビューで追いやすいよう、運用ルールを表で固定する。

### profile の意味
| profile | 解釈 | 使う設定 | 備考 |
| --- | --- | --- | --- |
| default | v1 grid | `pipeline.grid.*` / `pipeline.model_set` | 互換モード |
| custom | v2 groups | `pipeline.groups.*` + registry defaults | `plan.json` に groups 情報を記録 |

### groups.mode / custom.base
| mode | 意味 | 使いどころ |
| --- | --- | --- |
| none | グループを plan から除外 | 実験で完全停止 |
| default | registry の `default_enabled` を採用 | 既定運用 |
| custom | base + include/exclude で差分 | 追加/除外を明示 |

| custom.base | 意味 |
| --- | --- |
| default | `default_enabled` から開始 |
| none | 空集合から開始 |

- include/exclude は base 適用後に差分として処理する。
- unknown variant はエラー（レビュー時は include/exclude が実在するか確認する）。
- custom(include/exclude) は preprocess/train のみ解釈する。
- ensemble/leaderboard は `mode` による on/off のみ（variant 選択は持たない）。

### preprocess × model の展開
- preprocess variant を列挙し、各 preprocess から train job を展開する。
- `pipeline.builder.train_project_per_preprocess=true` の場合、Train の project を `<TrainGroup>/<preprocess_variant>` に階層化する。
- ensemble.enabled=true の場合、preprocess ごとに train_ensemble を 1 つ作成し、対象 preprocess の train だけを親にする。

### optional deps / SKIP の扱い
| 設定 | 値 | 効果 | 記録先 |
| --- | --- | --- | --- |
| `pipeline.groups.preprocess.custom.on_inapplicable` | skip/include/error | schema 非対応の前処理 | `plan.json` の `groups.preprocess.skipped_inapplicable` |
| `pipeline.groups.train.custom.on_missing_dependency` | skip/include/error | optional deps 未導入のモデル | `plan.json` の `groups.train.skipped_missing_dependencies` |

- skip は **plan から除外**し、実行対象に含めない。
- 実行後に SKIP が発生した場合は `run_summary.json` の各 entry に `status=skipped` と `reason/error` を残す。
- preprocess の missing dependency は既定で skip され、`groups.preprocess.skipped_missing_dependencies` に記録される。
- train_model を単独実行した場合も optional deps 欠如は `status=skipped` / `reason=missing_dependency` で記録される。

## partial failure（fail_policy）
pipeline は「SKIP/FAIL が混ざっても有用」という前提で、fail_policy を固定ルールとして運用する。

### fail_policy key
| key | 意味 | 既定 |
| --- | --- | --- |
| `allow_skipped` | SKIP を failure に含めるか | true |
| `allowed_failures` | 許容する failed 数（train_model） | 0 |
| `fail_fast` | allowed_failures 超過時に早期停止 | false |
| `min_successful_train_tasks` | 成功した train タスク数の最低条件 | 1 |

### 成否判定
- `successful_train_tasks >= min_successful_train_tasks` かつ
  `effective_failures <= allowed_failures`（effective_failures = failed + (skipped if allow_skipped=false)）
  なら SUCCESS
- それ以外は FAILED
- SUCCESS の場合でも skip/fail があれば `run_summary.json` に `degraded=true` を記録する

### run_summary.json（見方）
| 項目 | 意味 | 目安 |
| --- | --- | --- |
| `status` | success / failed | `plan_only` は success 扱い |
| `degraded` | skip/fail/limit の有無 | true の場合はレビュー必須 |
| `policy.fail_policy` | 実際に使われた fail_policy | config の最終値 |
| `policy.result.*` | success/failed/skipped の集計 | allow_skipped が反映される |
| `execution.*` | plan_only / planned_jobs / executed_jobs / skipped_due_to_policy / limit_exceeded | max_jobs/HPO 上限の影響を見る |
| `preprocess_variants` / `train_tasks` / `ensemble_tasks` | 各タスクの status/reason | SKIP の理由確認用 |
| `leaderboard` / `infer` | 代表タスクの status/reason | 下流工程の確認用 |

### 推奨値（例）
- 試験段階: `allow_skipped=true`, `allowed_failures=1-2`, `fail_fast=false`, `min_successful_train_tasks=1`
- 本番段階: `allow_skipped=false`, `allowed_failures=0`, `fail_fast=true`, `min_successful_train_tasks=1`

## pipeline v2 解釈（profile=custom）
- `pipeline.profile=custom` の場合のみ v2 解釈が有効になる（それ以外は `pipeline.grid.*` ベースの互換動作）。
- plan は `plan.json` として出力され、driver（local/controller）共通で利用する。
- `plan.json` には `profile` / `groups` / `limits` / `pipeline_limits` / `parallelism` が記録される。
- preprocess/train の groups には `candidates` / `variants` / `skipped_*` が記録される（optional deps の判定確認に使用）。
