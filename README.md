# Tabular Analysis Solution

`ml-solution-tabular-analysis` は、表形式データ向けの前処理、単体モデル学習、アンサンブル学習、比較評価、推論、ClearML 運用をまとめて扱う solution repository です。  
`ml_platform` を土台にしつつ、tabular ドメイン固有の処理と ClearML UI 運用を solution 側に閉じ込めています。

この README は、第三者が次の 3 つをすぐに把握できることを目的にしています。

- リポジトリ全体の構成とコードの立ち位置
- ローカル実行、ClearML logging、ClearML Pipeline Controller 実行の違い
- どのシーンでどのコマンドを使えばよいか

## 1. このリポジトリでできること

- CSV / Parquet からのデータ登録
- tabular 向け前処理レシピの作成と再利用
- 回帰 / 二値分類の単体モデル学習
- 複数モデルの leaderboard 比較
- 3 種のアンサンブル学習
  - `mean_topk`
  - `weighted`
  - `stacking`
- 学習済み bundle または ClearML Model Registry を使った推論
- ClearML Pipelines タブを正本にした pipeline template 管理
- ClearML Agent を使った controller / heavy model / default worker の分離実行

## 2. リポジトリの立ち位置

この solution repo は polyrepo の一部です。

- `ml_platform`
  - Hydra、ClearML、artifact、manifest、共通 runtime の土台
- `ml-solution-tabular-analysis`
  - tabular 向けの task 実装、model registry、ClearML UI 契約、運用 docs

solution 側の責務は「tabular の業務ロジック」と「ClearML 上での見え方を整えること」です。  
platform 側に寄せられるものは寄せつつ、solution 側は task ごとの処理を読みやすく保つ方針です。

## 3. 全体フロー

通常の回帰 / 分類フローは次です。

```text
dataset_register
  -> preprocess
  -> train_model (複数 variant)
  -> train_ensemble (optional)
  -> leaderboard
  -> infer (optional)
```

役割の整理:

- `dataset_register`
  - 入力データを検証し、ClearML Dataset と raw schema を確定する
- `preprocess`
  - split、feature type 判定、前処理 recipe を作る
- `train_model`
  - 単体モデルを学習し、metrics / preds / model bundle を出す
- `train_ensemble`
  - 既存の train 結果から ensemble を構築する
- `leaderboard`
  - 候補を比較して推奨モデルを選ぶ
- `infer`
  - 学習済み model bundle または registry model を使って推論する
- `pipeline`
  - 上記を一括で計画・実行する

## 4. ディレクトリ構成

リポジトリの主要構成は次です。

```text
conf/        Hydra 設定
src/         実装本体
tools/       補助ツール、template 管理、rehearsal、tests、agent assets
docs/        運用・設計ドキュメント
requirements/ pip fallback 用 requirements
work/        実行生成物の既定置き場
artifacts/   template plan などの一時生成物
```

特に重要な場所:

- `src/tabular_analysis/cli.py`
  - 全 task の入口
- `src/tabular_analysis/processes/`
  - task ごとの本体
- `src/tabular_analysis/registry/models.py`
  - model variant と optional dependency の正本
- `src/tabular_analysis/clearml/`
  - ClearML template / pipeline / hparams 契約
- `src/tabular_analysis/platform_adapter_*.py`
  - ClearML と solution の接続面
- `conf/task/`
  - task ごとの既定値
- `conf/group/model/`
  - model variant 定義
- `conf/pipeline/model_sets/`
  - `regression_all` のようなモデル集合
- `tools/clearml_templates/manage_templates.py`
  - ClearML template の `plan/apply/validate`
- `tools/rehearsal/run_pipeline_v2.py`
  - toy dataset での canonical 実行 runner
- `tools/clearml_agent/compose.yaml`
  - ClearML Agent の canonical Docker Compose

詳細は [docs/65_DEV_GUIDE_DIRECTORY_MAP.md](docs/65_DEV_GUIDE_DIRECTORY_MAP.md) を参照してください。

## 5. 前提環境

### 必須

- Python 3.10+
- Git
- `uv` または `pip`
- ClearML を使う場合は `clearml.conf` または `CLEARML_*` 環境変数

### 任意

- Docker / Docker Compose
  - ClearML Agent をコンテナで動かす場合
- FastAPI
  - serving API を使う場合
- LightGBM / XGBoost / CatBoost / TabPFN
  - 該当モデルを使う場合

## 6. 環境構築

### 6.1 ローカル開発の標準

```bash
uv sync --frozen
```

optional dependency を含める例:

```bash
uv sync --frozen --extra lightgbm --extra xgboost --extra catboost
uv sync --frozen --extra api
```

### 6.2 pip fallback

```bash
pip install -r requirements/base.txt
pip install -e .
```

### 6.3 ClearML 設定

PowerShell:

```powershell
$env:CLEARML_CONFIG_FILE="D:\Main_code\auto_ML-1\clearml.conf"
```

Bash:

```bash
export CLEARML_CONFIG_FILE=/path/to/clearml.conf
```

### 6.4 platform を横で開発したい場合

この repo 単体でも動きますが、platform 側も同時に読みたい場合は sibling checkout で管理すると追いやすいです。

```text
d:\tabular_clearml\
  ml_platform_v1-master\
  ml_taularanalysis_v1-master\
```

## 7. 入口コマンド

### 7.1 CLI

```bash
python -m tabular_analysis.cli task=<task>
```

設定確認だけしたい場合:

```bash
python -m tabular_analysis.cli task=pipeline --print-config
```

### 7.2 Rehearsal runner

```bash
python tools/rehearsal/run_pipeline_v2.py --execution local --task-type regression
```

### 7.3 ClearML template 管理

```bash
python tools/clearml_templates/manage_templates.py --plan --project-root LOCAL
python tools/clearml_templates/manage_templates.py --apply --project-root LOCAL
python tools/clearml_templates/manage_templates.py --validate --project-root LOCAL
```

## 8. 実行モード

`run.clearml.execution` で実行形態を切り替えます。

| mode | 用途 | ClearML 上の挙動 |
| --- | --- | --- |
| `local` | ローカル確認 | ClearML なし |
| `logging` | UI 契約確認 | ローカル実行しつつ ClearML に記録 |
| `agent` | 単一 task の remote 実行 | task を queue へ投げる |
| `clone` | template task を元に remote 実行 | template clone |
| `pipeline_controller` | operator 向け本番運用 | visible pipeline template を clone して controller 実行 |

現在の operator 正本は `pipeline_controller` です。  
ClearML Pipelines タブ上の visible template を確認、clone、実行する流れを想定しています。

詳細は [docs/10_OPERATION_MODES.md](docs/10_OPERATION_MODES.md) を参照してください。

## 9. ClearML 運用の基本

### 9.1 template の正本

ClearML template は `manage_templates.py` で管理します。  
現在は child task template に加えて、visible pipeline template も first-class に管理しています。

主な visible pipeline template:

- `pipeline`
  - preprocess + single-model train + leaderboard
- `train_model_full`
  - preprocess + single-model train
- `train_ensemble_full`
  - preprocess + single-model train + 3 ensemble + leaderboard

visible pipeline template は ClearML 上の次 project に置かれます。

```text
<project_root>/TabularAnalysis/Pipelines
```

### 9.2 queue の役割

現在の canonical queue 分割:

- `controller`
  - pipeline controller 専用
- `default`
  - preprocess、軽量 train、leaderboard、ensemble、infer
- `heavy-model`
  - `catboost`、`xgboost`

### 9.3 Agent の canonical 起動方法

```bash
docker compose -f tools/clearml_agent/compose.yaml up -d --build
```

Compose は次を前提にしています。

- Docker named volume の shared `/root/.clearml`
- `UV_CACHE_DIR=/root/.clearml/uv-cache`

詳細は [docs/68_CLEARML_AGENT_TROUBLESHOOTING.md](docs/68_CLEARML_AGENT_TROUBLESHOOTING.md) を参照してください。

## 10. 機能一覧

### 10.1 データ登録

```bash
python -m tabular_analysis.cli task=dataset_register \
  data.dataset_path=/path/to/data.csv \
  data.target_column=target \
  run.clearml.enabled=false
```

### 10.2 前処理のみ

```bash
python -m tabular_analysis.cli task=preprocess \
  data.dataset_path=/path/to/data.csv \
  data.target_column=target \
  preprocess.variant=stdscaler_ohe \
  run.clearml.enabled=false
```

### 10.3 単体モデル学習

```bash
python -m tabular_analysis.cli task=train_model \
  train.inputs.preprocess_run_dir=/path/to/preprocess_run \
  group/model=ridge \
  run.clearml.enabled=false
```

### 10.4 アンサンブル学習

```bash
python -m tabular_analysis.cli task=train_ensemble \
  run.usecase_id=my_regression_case \
  preprocess.variant=stdscaler_ohe \
  ensemble.enabled=true \
  ensemble.methods=[mean_topk,weighted,stacking] \
  run.clearml.enabled=false
```

### 10.5 leaderboard

```bash
python -m tabular_analysis.cli task=leaderboard \
  leaderboard.train_run_dirs=[/path/to/train1,/path/to/train2] \
  run.clearml.enabled=false
```

### 10.6 推論

single inference:

```bash
python -m tabular_analysis.cli task=infer \
  infer.mode=single \
  infer.model_id=/path/to/model_bundle.joblib \
  infer.single.input_json='{"num1":1.0,"num2":2.0,"cat":"a"}'
```

batch inference:

```bash
python -m tabular_analysis.cli task=infer \
  infer.mode=batch \
  infer.model_id=<CLEARML_REGISTRY_MODEL_ID> \
  infer.batch.inputs_path=/path/to/batch.csv
```

### 10.7 Serving API

```bash
uv sync --frozen --extra api
uvicorn tabular_analysis.serve.app:app --reload
```

## 11. 実行シーン別の推奨手順

### 11.1 まずローカルで動作確認したい

```bash
python tools/rehearsal/run_pipeline_v2.py \
  --execution local \
  --task-type regression \
  --preprocess stdscaler_ohe \
  --models ridge,elasticnet
```

### 11.2 ClearML にログだけ残したい

```bash
python tools/rehearsal/run_pipeline_v2.py \
  --execution logging \
  --task-type regression \
  --preprocess stdscaler_ohe \
  --models ridge,elasticnet \
  --project-root LOCAL
```

### 11.3 operator と同じ visible pipeline template を使いたい

template を同期:

```bash
python tools/clearml_templates/manage_templates.py --apply --project-root LOCAL
python tools/clearml_templates/manage_templates.py --validate --project-root LOCAL
```

controller 実行:

```bash
python -m tabular_analysis.cli task=pipeline \
  run.clearml.enabled=true \
  run.clearml.execution=pipeline_controller \
  run.clearml.project_root=LOCAL \
  data.raw_dataset_id=<RAW_DATASET_ID> \
  +pipeline.preprocess_variant=stdscaler_ohe \
  +pipeline.model_set=regression_all
```

### 11.4 3 種アンサンブルまで含めて回したい

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

または ClearML UI の Pipelines タブから `train_ensemble_full` template を clone して実行します。

### 11.5 UI 契約を監査したい

```bash
python tools/tests/rehearsal_verify_clearml_ui.py \
  --usecase-id <USECASE_ID> \
  --project-root LOCAL
```

## 12. モデルと model set

標準の回帰 full set は `conf/pipeline/model_sets/regression_all.yaml` です。  
現在の canonical 13 モデル:

- `catboost`
- `elasticnet`
- `extra_trees`
- `gaussian_process`
- `gradient_boosting`
- `knn`
- `lasso`
- `lgbm`
- `linear_regression`
- `mlp`
- `random_forest`
- `ridge`
- `xgboost`

`svr` は標準 full set から除外されています。必要な場合は明示指定してください。

optional dependency は model 単位で分離しています。

- `lgbm` -> `lightgbm`
- `xgboost` -> `xgboost`
- `catboost` -> `catboost`
- `tabpfn` -> `tabpfn`

## 13. 出力物

各 task は共通して次を出します。

- `config_resolved.yaml`
- `out.json`
- `manifest.json`

代表的な成果物:

- `dataset_register`
  - `schema.json`, `preview.csv`
- `preprocess`
  - `preprocess_bundle.*`, `recipe.json`, `split.json`, `summary.md`
- `train_model`
  - `model_bundle.*`, `metrics.json`, `preds_valid.parquet`
- `train_ensemble`
  - `model_bundle.joblib`, `ensemble_spec.json`, `metrics.json`
- `leaderboard`
  - `leaderboard.csv`, `recommendation.json`
- `pipeline`
  - `pipeline_run.json`, `report.md`, `report.json`, `report_links.json`

## 14. ドキュメントの読み順

まずはここから読むのがおすすめです。

1. [docs/INDEX.md](docs/INDEX.md)
2. [docs/02_ARCHITECTURE.md](docs/02_ARCHITECTURE.md)
3. [docs/05_PROCESS_CATALOG.md](docs/05_PROCESS_CATALOG.md)
4. [docs/10_OPERATION_MODES.md](docs/10_OPERATION_MODES.md)
5. [docs/65_DEV_GUIDE_DIRECTORY_MAP.md](docs/65_DEV_GUIDE_DIRECTORY_MAP.md)
6. [docs/67_REHEARSAL_COMMANDS.md](docs/67_REHEARSAL_COMMANDS.md)
7. [docs/81_CLEARML_TEMPLATE_POLICY.md](docs/81_CLEARML_TEMPLATE_POLICY.md)
8. [docs/82_CLEARML_PROJECT_LAYOUT.md](docs/82_CLEARML_PROJECT_LAYOUT.md)

## 15. テストと確認

quick verify:

```bash
python tools/tests/verify_all.py --quick
```

docs path check:

```bash
python tools/tests/check_docs_paths.py --repo .
```

template contract check:

```bash
python tools/tests/test_template_specs.py
python tools/tests/test_clearml_runtime_contracts.py
```

## 16. 補足

- ClearML の operator 向け正本は visible pipeline template です。
- runtime の bootstrap は task-time install を維持しますが、`uv` cache 共有前提です。
- `work/` と `artifacts/` は実行生成物の置き場です。コミット前に中身を確認してください。

