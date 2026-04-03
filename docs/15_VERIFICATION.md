# 15 Verification

## 目的

このドキュメントは、`ml-solution-tabular-analysis` の検証手順をまとめたものです。  
ローカルだけの確認、ClearML logging の確認、release 前の包括確認を分けて考えます。

## 推奨の確認順

1. docs / static check
2. quick verify
3. task 単体 smoke
4. local pipeline smoke
5. ClearML logging / template contract

## 1. 最低限の quick verify

```bash
python tools/tests/verify_all.py --quick
```

これで主に次を確認します。

- 主要 smoke
- docs path
- contract test
- optional model の存在確認
- serving import の基本確認

## 2. docs / static check

```bash
python tools/tests/check_docs_paths.py --repo .
python -m compileall -q src
```

## 3. task 単体 smoke

### local pipeline まで

```bash
python tools/tests/smoke_local.py --until pipeline
```

### 分類 smoke

```bash
python tools/tests/smoke_classification.py --until leaderboard
```

### 回帰モデル単体

```bash
python tools/tests/smoke_train_regression_model.py --model random_forest --expect-feature-importance
```

### build-only

```bash
python tools/tests/smoke_model_build_only.py --task-type regression --models knn,gaussian_process,mlp
python tools/tests/smoke_model_build_only.py --task-type classification --models svc,knn,gaussian_process,mlp
```

## 4. optional model 確認

```bash
python tools/tests/check_optional_models.py --models lgbm,xgboost,catboost,tabpfn
```

optional dependency を入れて検証したい場合:

```bash
uv sync --frozen --extra lightgbm --extra xgboost --extra catboost
uv sync --frozen --extra tabpfn
```

## 5. report / plot / HPO 関連

```bash
python tools/tests/smoke_report.py
python tools/tests/smoke_plots.py
python tools/tests/smoke_hpo.py
```

## 6. ClearML 契約確認

### template / runtime contract

```bash
python tools/tests/test_template_specs.py
python tools/tests/test_clearml_runtime_contracts.py
```

### UI 監査

```bash
python tools/tests/rehearsal_verify_clearml_ui.py --usecase-id <USECASE_ID> --project-root LOCAL
```

## 7. full verify

```bash
python tools/tests/verify_all.py --full
```

使いどころ:

- release 前
- 大きな refactor 後
- ClearML 運用契約を変えた後

## 8. release 前の推奨セット

```bash
python tools/tests/check_docs_paths.py --repo .
python tools/tests/test_template_specs.py
python tools/tests/test_clearml_runtime_contracts.py
python tools/tests/verify_all.py --quick
```

必要に応じて:

```bash
python tools/tests/verify_all.py --full
```

## 9. 失敗時の見方

- import failure
  - dependency か `PYTHONPATH`
- optional model missing
  - extras 未導入
- ClearML contract failure
  - template / project / tags / queue の不整合
- smoke failure
  - `out.json` と artifact を確認

## 関連ドキュメント

- [16_OPERATIONS_RUNBOOK.md](16_OPERATIONS_RUNBOOK.md)
- [67_REHEARSAL_COMMANDS.md](67_REHEARSAL_COMMANDS.md)
- [69_CLEARML_TROUBLESHOOTING.md](69_CLEARML_TROUBLESHOOTING.md)


