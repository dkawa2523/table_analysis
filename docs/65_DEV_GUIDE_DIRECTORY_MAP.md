# 65 Dev Guide Directory Map

このファイルは、実装者向けの「どこを見ればよいか」の地図です。  
README が利用者向けなら、こちらは保守・改修向けです。

## 1. まず何を見るべきか

### task を追いたい

1. `src/tabular_analysis/cli.py`
2. `conf/task/<task>.yaml`
3. `src/tabular_analysis/processes/<task>.py`
4. 関連する `registry/`, `common/`, `clearml/`

### ClearML の表示や template を直したい

1. `conf/clearml/templates.yaml`
2. `tools/clearml_templates/manage_templates.py`
3. `src/tabular_analysis/clearml/`
4. `src/tabular_analysis/platform_adapter_pipeline.py`
5. `src/tabular_analysis/platform_adapter_task*.py`

### model を増やしたい

1. `conf/group/model/<variant>.yaml`
2. `src/tabular_analysis/registry/models.py`
3. `conf/pipeline/model_sets/regression_all.yaml` など

### 前処理を変えたい

1. `conf/group/preprocess/`
2. `src/tabular_analysis/processes/preprocess.py`
3. `src/tabular_analysis/feature_engineering/`

## 2. top-level ディレクトリ

## `src/`

実装本体です。最重要ディレクトリです。

## `conf/`

Hydra 設定です。task、model、pipeline、ClearML、execution policy を定義します。

## `tools/`

template 管理、rehearsal、tests、agent assets など、運用を支える補助ツールです。

## `docs/`

設計、契約、運用手順の正本です。

## `requirements/`

pip fallback 用の依存定義です。

## `work/`

実行生成物の既定置き場です。

## `artifacts/`

template plan など一時生成物の置き場です。

## 3. `src/tabular_analysis/` の詳細

## `cli.py`

- solution の単一 entrypoint
- `task.name` に応じて process runner を選ぶ

## `processes/`

task の本体です。

- `dataset_register.py`
- `preprocess.py`
- `train_model.py`
- `train_ensemble.py`
- `leaderboard.py`
- `infer.py`
- `pipeline.py`
- `retrain.py`
- `lifecycle.py`
  - 共通 runtime bootstrapping

### ここを触るときの考え方

- task 固有の業務ロジックは `processes/`
- 共通化できる pure helper は `common/` や `registry/`
- ClearML 固有の処理は `clearml/` や `platform_adapter_*`

## `registry/`

variant や計算ロジックの正本です。

- `models.py`
  - model variant、task_type、optional extras
- `metrics.py`
  - metric helper

## `common/`

純粋関数や shared contract を置く場所です。

- feature type
- probability helper
- drift helper
- ClearML bootstrap helper
- model reference helper

## `clearml/`

ClearML 契約の中心です。

- template lookup
- pipeline template helper
- hyperparameter section helper
- visible pipeline controller 契約

## `platform_adapter_*.py`

ClearML との接続面です。  
top-level shim は廃止し、family module に分割しています。

- `platform_adapter_core.py`
- `platform_adapter_task.py`
- `platform_adapter_task_ops.py`
- `platform_adapter_pipeline.py`
- `platform_adapter_model.py`
- `platform_adapter_artifacts.py`
- `platform_adapter_clearml_env.py`

## `feature_engineering/`

tabular 特有の前処理ロジックです。

## `metrics/`, `reporting/`, `quality/`, `viz/`

学習結果の可視化、品質チェック、report 生成の層です。

## `monitoring/`, `uncertainty/`

drift や uncertainty の補助ロジックです。

## `serve/`

任意の FastAPI serving 用です。

## `ops/`

運用補助です。

- ClearML identity
- data quality

## 4. `conf/` の詳細

## `conf/task/`

task ごとの既定 config です。  
まずここを見ると task の既定フローが分かります。

## `conf/group/model/`

model variant 定義です。

例:

- `ridge.yaml`
- `lgbm.yaml`
- `xgboost.yaml`
- `catboost.yaml`

## `conf/group/preprocess/`

preprocess variant 定義です。現在の主力は `stdscaler_ohe.yaml` です。

## `conf/pipeline/model_sets/`

pipeline で使う model 集合です。  
現在の標準回帰 full set は `regression_all.yaml` です。

## `conf/clearml/`

template、project layout、UI 契約に関わる設定です。

重要ファイル:

- `templates.yaml`
- `project_layout.yaml`

## `conf/exec_policy/`

queue、heavy-model routing、parallelism、limit などの実行ポリシーです。

## `conf/run/`

run 共通の設定です。`run.clearml.*` などがまとまっています。

## `conf/data/`, `conf/eval/`, `conf/ensemble/`, `conf/leaderboard/`

task 周辺の補助設定です。

## 5. `tools/` の詳細

## `tools/clearml_templates/`

template の `plan/apply/validate` を担当します。  
operator 運用で最重要です。

## `tools/rehearsal/`

toy dataset を使った canonical runner です。

- `run_pipeline_v2.py`

## `tools/tests/`

smoke test、contract test、UI verify を置きます。

よく使うもの:

- `verify_all.py`
- `check_docs_paths.py`
- `rehearsal_verify_clearml_ui.py`
- `test_template_specs.py`
- `test_clearml_runtime_contracts.py`

## `tools/clearml_agent/`

Agent の canonical Dockerfile / Compose です。

## 6. 実装観点ごとの参照先

### 「task は通るが UI が変」

- `conf/clearml/templates.yaml`
- `src/tabular_analysis/clearml/pipeline_templates.py`
- `src/tabular_analysis/processes/pipeline.py`
- `src/tabular_analysis/platform_adapter_pipeline.py`

### 「heavy model の queue 分岐を変えたい」

- `conf/exec_policy/base.yaml`
- `src/tabular_analysis/processes/pipeline.py`
- `tools/clearml_agent/compose.yaml`

### 「optional dependency を減らしたい」

- `pyproject.toml`
- `src/tabular_analysis/registry/models.py`
- `src/tabular_analysis/common/clearml_bootstrap.py`
- `tools/clearml_entrypoint.py`

### 「report の内容を変えたい」

- `src/tabular_analysis/reporting/`
- `src/tabular_analysis/processes/leaderboard.py`
- `src/tabular_analysis/processes/pipeline.py`

## 7. 併読するとよい docs

- [02_ARCHITECTURE.md](02_ARCHITECTURE.md)
- [05_PROCESS_CATALOG.md](05_PROCESS_CATALOG.md)
- [10_OPERATION_MODES.md](10_OPERATION_MODES.md)
- [16_OPERATIONS_RUNBOOK.md](16_OPERATIONS_RUNBOOK.md)
- [67_REHEARSAL_COMMANDS.md](67_REHEARSAL_COMMANDS.md)
- [81_CLEARML_TEMPLATE_POLICY.md](81_CLEARML_TEMPLATE_POLICY.md)
- [82_CLEARML_PROJECT_LAYOUT.md](82_CLEARML_PROJECT_LAYOUT.md)

