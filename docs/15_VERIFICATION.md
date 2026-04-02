# 15 Verification（検証コマンド集）

このドキュメントは、`ml-solution-tabular-analysis` を **ローカルモード（ClearML無効）** で検証するためのコマンド集です。

- 基本的に `requirements/base.txt` をインストール済みであることを前提とします。
- ここでの検証は「最低限の契約（config_resolved.yaml / out.json / manifest.json）」が保たれているかを中心に確認します。

---

## A) 統合コマンド（推奨）

日常開発や CI では quick を使います（主要スモーク + 多クラス + 高カーディナリティ）。

```bash
python tools/tests/verify_all.py --quick
```

リリース前の手動確認では full を使います（quick + 追加スモーク: 不確かさ / 指標CI / 監視 / ガバナンス / スケール / Serving など）。

```bash
python tools/tests/verify_all.py --full
```

full では `tools/tests/smoke_train_regression_model.py` が存在する場合のみ実行します。fastapi など optional dependency が無いテストは自動 skip されます。

---

## B) ローカル必須チェック（個別実行）

### 1) 構文チェック
```bash
python -m compileall -q src
```

### 2) 回帰E2Eスモーク（pipeline まで）
```bash
python tools/tests/smoke_local.py --until pipeline
```

### 3) 分類E2Eスモーク（leaderboard まで）
```bash
python tools/tests/smoke_classification.py --until leaderboard
```

### 4) 代表モデルのスモーク（回帰）
```bash
python tools/tests/smoke_train_regression_model.py --model random_forest --expect-feature-importance
```

### 5) build-only チェック（重いモデルは学習せず構築のみ）
```bash
python tools/tests/smoke_model_build_only.py --task-type regression --models knn,svr,gaussian_process,mlp
python tools/tests/smoke_model_build_only.py --task-type classification --models svc,knn,gaussian_process,mlp
```

### 6) optional model の健全性（依存が無い場合は graceful error を確認）
```bash
python tools/tests/check_optional_models.py --models lgbm,xgboost,catboost,tabpfn
```

### 7) HPO / report / plots
```bash
python tools/tests/smoke_hpo.py
python tools/tests/smoke_report.py
python tools/tests/smoke_plots.py
```

### 8) CI workflow 設定のスモーク
```bash
python tools/tests/smoke_ci_config.py
```

---

## C) optional deps を入れて「実際に学習」まで確認したい場合

外部モデル（LightGBM/XGBoost/CatBoost/TabPFN）を実際に動かす場合は extras を入れます。

```bash
pip install -e ".[models]"
# TabPFN も試す場合
pip install -e ".[tabpfn]"
```

その後、モデル実在チェックを再実行します。
```bash
python tools/tests/check_optional_models.py --models lgbm,xgboost,catboost,tabpfn
```

---

## D) ClearML 有効での確認（環境が整っている場合のみ）

ClearML の設定（`clearml.conf` / 環境変数 / API key 等）が完了している場合、pipeline を ClearML 有効で実行できます。

```bash
python -m tabular_analysis.cli task=pipeline \
  run.clearml.enabled=true run.clearml.execution=logging \
  data.raw_dataset_id=<RAW_DATASET_ID>
```

---

## E) T029〜T039 追加機能のスモーク（verify_all --full で実行）

個別に確認したい場合は以下を実行します（repo 直下で実行する想定）。

### 1) 多クラス / 不均衡 / 高カーディナリティ
```bash
python tools/tests/smoke_multiclass.py
python tools/tests/smoke_imbalance.py
python tools/tests/smoke_high_card_cat.py
```

### 2) 不確かさ / 指標CI
```bash
python tools/tests/smoke_uncertainty.py
python tools/tests/smoke_metric_ci.py
```

### 3) 校正 / Decision Summary
```bash
python tools/tests/smoke_calibration.py
python tools/tests/smoke_decision_summary.py
```

### 4) 監視 / スケール / Serving
```bash
python tools/tests/smoke_drift_enhanced.py
python tools/tests/smoke_batch_chunked.py
python tools/tests/smoke_serve_import.py
```

Serving は fastapi が入っていない場合に自動 skip されます。

---

## F) まとめ

基本は `verify_all.py` を使い、必要に応じて個別スモークを追加実行してください。
