# 86 ClearML Internals For Developers

このファイルは、developer が current ClearML pipeline 実装を安全に変更するための内部メモです。  
operator 向けの手順は [17_OPERATOR_QUICKSTART.md](17_OPERATOR_QUICKSTART.md) と [16_OPERATIONS_RUNBOOK.md](16_OPERATIONS_RUNBOOK.md) を参照してください。

## 1. まず持つべきメンタルモデル

現在の ClearML contract は 4 層です。

- seed pipeline
  - `<project_root>/TabularAnalysis/.pipelines/<profile>`
- actual run controller
  - `<project_root>/TabularAnalysis/Pipelines/Runs/<usecase_id>`
- child task
  - `<project_root>/TabularAnalysis/Runs/<usecase_id>/<group>`
- step template
  - `<project_root>/TabularAnalysis/Templates/Steps/<group>`

重要:

- operator の正規入口は seed card の `NEW RUN`
- `NEW PIPELINE` は current contract の対象外
- 実編集の正本は `Hyperparameters`
- `Configuration > OperatorInputs` は read-only mirror

## 2. source of truth

### project / identity / naming

- [clearml_identity.py](../src/tabular_analysis/ops/clearml_identity.py)
  - project path
  - tag / property contract
  - `Runs/<usecase_id>/<group>` と `.pipelines/<profile>` の命名
  - usecase 自動採番

### seed lookup / base task resolution

- [pipeline_templates.py](../src/tabular_analysis/clearml/pipeline_templates.py)
  - active seed lookup
  - explicit `run.clearml.pipeline.template_task_id` fallback

### pipeline runtime normalization

- [pipeline_support.py](../src/tabular_analysis/processes/pipeline_support.py)
  - built-in seed profiles
  - editable whitelist
  - `OperatorInputs` mirror
  - placeholder `raw_dataset_id` fail-fast
  - `run.usecase_id` 自動採番と clone 正規化
  - summary / report 向け集計

- [pipeline.py](../src/tabular_analysis/processes/pipeline.py)
  - orchestration
  - controller plan
  - local / clearml / current-task 実行分岐

### task mutation

- [platform_adapter_task_ops.py](../src/tabular_analysis/platform_adapter_task_ops.py)
  - task parameter read / write / reset の正本
  - encoded key cleanup の正本

- [platform_adapter_core.py](../src/tabular_analysis/platform_adapter_core.py)
  - 薄い wrapper
  - unrelated local diff があるため、大きな cleanup は慎重に行う

### entrypoint override normalization

- [clearml_entrypoint.py](../tools/clearml_entrypoint.py)
  - `%2E` / `%2F` を plain dotted key へ正規化
  - Hydra に渡す override を canonical 化

### live seed refresh / cleanup

- [manage_templates.py](../tools/clearml_templates/manage_templates.py)
  - seed apply / validate
  - stale seed / probe の deprecate
  - `_DeprecatedPipelines` への退避

## 3. runtime 境界

### 3.1 seed apply 時

`manage_templates --apply` は、step template を同期したうえで built-in profile ごとの seed pipeline を更新します。  
この時点で seed は:

- `task_kind:seed`
- `.pipelines/<profile>`
- profile 固定の DAG
- placeholder を含みうる `OperatorInputs`

を持ちます。

### 3.2 UI `NEW RUN` clone 時

ClearML UI が作るのは seed の clone です。  
この clone は最初から正しい run metadata を持つとは限りません。

### 3.3 entrypoint 時

`tools/clearml_entrypoint.py` が最初の正規化境界です。

- encoded key を plain dotted key に戻す
- duplicate key を潰す
- Hydra に壊れた `%2E` key を流さない

### 3.4 pipeline runtime 時

`pipeline_support.py` / `pipeline.py` が current task を actual run shape に正規化します。

- `task_kind:seed` -> `task_kind:run`
- project を `.pipelines/<profile>` から `Pipelines/Runs/<usecase_id>` へ更新
- `usecase:<id>` tags / properties を actual run 用に更新
- `OperatorInputs` を current values で再生成
- placeholder `data.raw_dataset_id` を fail-fast

### 3.5 child task launch 時

child task は step template を clone したあと、runtime identity を rebuild します。

- child project は `Runs/<usecase_id>/<group>`
- template 用 stale tag は除去
- `process:*`, `usecase:*`, `grid:*` など runtime tag を再付与

## 4. 変更時に守るべき contract

### operator contract

- operator は seed card から `NEW RUN` する
- 実編集は `Hyperparameters`
- `OperatorInputs` は mirror
- seed card の placeholder は正常
- actual run では placeholder 不可

### layout contract

- seed: `.pipelines/<profile>`
- run controller: `Pipelines/Runs/<usecase_id>`
- child: `Runs/<usecase_id>/<group>`
- step template: `Templates/Steps/<group>`

### queue contract

queue 正本は `exec_policy.queues.*` です。

- controller: `controller`
- light child: `default`
- heavy model child: `heavy-model`

`run.clearml.queue_name` は child routing の正本ではありません。

### selection contract

fixed DAG を profile が決め、subset は selection key で表します。

- `pipeline.selection.enabled_preprocess_variants`
- `pipeline.selection.enabled_model_variants`
- `ensemble.selection.enabled_methods`

inactive step は child task を作らず、controller summary/report に `disabled_by_selection` として残します。

## 5. legacy fallback の扱い

次は current operator flow の正本ではありません。

- `run.clearml.template_usecase_id`
- `run.clearml.pipeline.template_task_id`

扱い:

- runtime の read fallback
- helper の explicit override
- docs では `legacy read fallback only` と明記する

新規コードでこれらを source of truth に戻さないでください。

## 6. safe change checklist

1. seed / run / child / template の 4 層が崩れていないか
2. `.pipelines/<profile>` seed card が見えるか
3. `NEW RUN` 後に `OperatorInputs` と `Hyperparameters` の役割が崩れていないか
4. actual run が `Pipelines/Runs/<usecase_id>` と `Runs/<usecase_id>/*` に切れているか
5. 新規 run に `%2E` key が流れていないか
6. `manage_templates --apply --validate` が通るか
7. docs と tests を同じ batch で更新したか

## 7. どの docs を見るか

### operator 向け

- [17_OPERATOR_QUICKSTART.md](17_OPERATOR_QUICKSTART.md)
- [16_OPERATIONS_RUNBOOK.md](16_OPERATIONS_RUNBOOK.md)
- [55_CLEARML_UI_CHECKLIST.md](55_CLEARML_UI_CHECKLIST.md)
- [69_CLEARML_TROUBLESHOOTING.md](69_CLEARML_TROUBLESHOOTING.md)

### contract / internals 向け

- [03_CLEARML_UI_CONTRACT.md](03_CLEARML_UI_CONTRACT.md)
- [52_CLEARML_PIPELINE_CONTROLLER_CONTRACT.md](52_CLEARML_PIPELINE_CONTROLLER_CONTRACT.md)
- [53_CLEARML_HYPERPARAMETERS_CONTRACT.md](53_CLEARML_HYPERPARAMETERS_CONTRACT.md)
- [61_CLEARML_HPARAMS_SECTIONS.md](61_CLEARML_HPARAMS_SECTIONS.md)
- [63_CLEARML_PIPELINES_VISIBILITY.md](63_CLEARML_PIPELINES_VISIBILITY.md)
- [81_CLEARML_TEMPLATE_POLICY.md](81_CLEARML_TEMPLATE_POLICY.md)
- [82_CLEARML_PROJECT_LAYOUT.md](82_CLEARML_PROJECT_LAYOUT.md)
- [87_CLEARML_PIPELINE_WORKFLOW_DETAILS.md](87_CLEARML_PIPELINE_WORKFLOW_DETAILS.md)
