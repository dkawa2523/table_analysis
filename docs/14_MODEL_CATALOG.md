# Model Catalog（Tabular Analysis）

このドキュメントは ml-solution-tabular-analysis における **利用可能モデル一覧（目標：回帰/分類）** と、
依存関係（optional dependency）を整理したものです。

> 目的:
> - 開発者が conf/group/model/*.yaml を追加するときに迷わない
> - 業務投入で「モデルが選べない/依存が足りない」事故を防ぐ
> - ClearML 上の比較可能性（task_type / metric / split）を崩さない

---

## 1. モデル一覧（検討対象）

| モデル名（例） | 回帰/分類 | ライブラリ更新目安（最小要求） | 特徴 | メリット | デメリット | 使いどころ |
|---|---|---|---|---|---|---|
| `LinearRegression` | 回帰 | `scikit-learn>=1.3` | 線形（正則化なし） | 高速・基準値になりやすい | 非線形に弱い | まずベースライン |
| `Ridge` | 回帰/分類 | `scikit-learn>=1.3` | 線形 + L2正則化 | 安定・多重共線性に強い | 強い非線形は表現しにくい | 高次元のベースライン |
| `Lasso` | 回帰 | `scikit-learn>=1.3` | 線形 + L1正則化 | 特徴量選択（疎）になりやすい | 相関が強い特徴量が多いと不安定 | 特徴量を絞りたい |
| `ElasticNet` | 回帰 | `scikit-learn>=1.3` | 線形 + L1/L2 | Lassoより安定しやすい | 正則化係数の調整が必要 | 相関あり + 特徴選択 |
| `KNeighbors` (`KNN`) | 回帰/分類 | `scikit-learn>=1.3` | 近傍ベース | 学習が軽い・非線形に対応 | 推論が遅い/スケーリングに敏感 | 小規模・局所パターン |
| `SVR` | 回帰 | `scikit-learn>=1.3` | カーネルSVR | 小〜中規模で強いことがある | 大規模で遅い/パラメータ感度 | 非線形・データ少なめ |
| `RandomForest` | 回帰/分類 | `scikit-learn>=1.3` | 決定木のバギング | 頑健・扱いやすい | GBDT系より精度が伸びにくいことがある | まず堅い選択肢 |
| `ExtraTrees` | 回帰/分類 | `scikit-learn>=1.3` | ランダム性を強めた木 | 高速・過学習しにくいことがある | 精度のブレ/モデルが大きくなりがち | 速さ重視の木モデル |
| `GradientBoosting` | 回帰/分類 | `scikit-learn>=1.3` | sklearnのGBDT | 標準構成でRFより高精度になりやすい | 大規模だと遅い/機能は控えめ | 依存追加なしでGBDT |
| `GaussianProcess` | 回帰/分類 | `scikit-learn>=1.3` | ガウス過程 | 不確かさ推定が可能 | 計算が重い（データ数に弱い） | 超小規模・検証用途 |
| `MLP` | 回帰/分類 | `scikit-learn>=1.3` | sklearnのNN | 非線形表現が可能 | 収束やスケール調整が難しい | 木系が合わないとき |
| `LogisticRegression` | 分類 | `scikit-learn>=1.3` | 線形（分類） | 高速・解釈しやすい | 非線形に弱い | 線形ベースライン |
| `SVC` (`SVM`) | 分類 | `scikit-learn>=1.3` | カーネルSVM | 小〜中規模で強いことがある | 大規模で遅い/スケーリングに敏感 | 非線形・データ少なめ |
| `LightGBM`（要: `lightgbm`） | 回帰/分類 | `lightgbm>=4.0` | 高速GBDT | 定番・高精度になりやすい | パラメータが多い/過学習注意 | まず試す第一候補 |
| `XGBoost`（要: `xgboost`） | 回帰/分類 | `xgboost>=2.0` | 堅牢GBDT | 実績が多く安定しやすい | 設定が多い/学習が重め | LightGBMと比較 |
| `CatBoost`（要: `catboost`） | 回帰/分類 | `catboost>=1.2` | カテゴリに強いGBDT | カテゴリ列が多いと強いことがある | 学習が重め/依存追加 | カテゴリが多いデータ |
| `TabPFN`（要: `tabpfn` + 学習済み重み） | 回帰/分類 | `tabpfn>=0.1` | 事前学習モデル | 小規模データで強いことがある | 重みが無い場合は失敗として記録（自動DLも可） | データ少なめで試す |

---


---

## 1.1 conf での variant 名（推奨）

実装では、Hydra の `group/model=<variant>` でモデルを選びます。  
表の “モデル名” と conf のファイル名/variant 名の対応は次を推奨します（snake_case）。

| variant（group/model=...） | 表のモデル名 | 備考 |
|---|---|---|
| `linear_regression` | LinearRegression | 回帰のみ |
| `ridge` | Ridge | `class_path` dict で回帰/分類切替（RidgeClassifier） |
| `lasso` | Lasso | 回帰のみ |
| `elasticnet` | ElasticNet | 回帰のみ |
| `knn` | KNeighbors | `class_path` dict で回帰/分類切替 |
| `svr` | SVR | 回帰のみ（分類は `svc`） |
| `svc` | SVC | 分類のみ（回帰は `svr`） |
| `random_forest` | RandomForest | `class_path` dict で回帰/分類切替 |
| `extra_trees` | ExtraTrees | `class_path` dict で回帰/分類切替 |
| `gradient_boosting` | GradientBoosting | `class_path` dict で回帰/分類切替 |
| `gaussian_process` | GaussianProcess | `class_path` dict で回帰/分類切替 |
| `mlp` | MLP | `class_path` dict で回帰/分類切替 |
| `logistic_regression` | LogisticRegression | 分類のみ |
| `lgbm` | LightGBM | optional |
| `xgboost` | XGBoost | optional |
| `catboost` | CatBoost | optional |
| `tabpfn` | TabPFN | optional |

> NOTE:
> - `svr` / `svc` は `eval.task_type` に応じて SVR/SVC が選択される
> - 運用上は `svr`=回帰、`svc`=分類 を推奨

## 1.2 線形回帰系モデルの注意点

- `linear_regression`: 正則化なし。多重共線性が強い場合は `ridge` / `lasso` / `elasticnet` と比較する。
- `lasso` / `elasticnet`: 収束警告が出る場合は `max_iter` を増やす。標準化済み特徴（`stdscaler_ohe`）が安定。
- `elasticnet`: `l1_ratio` で L1/L2 のバランスを調整。Lasso が不安定なときの比較に向く。

## 1.3 非線形モデルの使いどころ/計算コスト/推奨データサイズ

| モデル | 使いどころ | 計算コスト注意 | 推奨データサイズ（目安） |
|---|---|---|---|
| `KNN` | 局所パターンや類似検索寄りのタスク | 学習は軽いが推論が重い（全件探索） | 〜数万行 |
| `SVR` / `SVC` | 非線形 + 少量データの比較 | 計算が重い（概ね O(n^2) 以上） | 〜2万行 |
| `GaussianProcess` | 検証用途・不確かさ推定 | 計算/メモリが非常に重い（O(n^3)） | 〜2,000行 |
| `MLP` | 木系が合わない場合の非線形表現 | ハイパーパラメータ依存・収束コスト | 1万〜数十万行 |

- 非線形モデルはスケーリングに敏感。`stdscaler_ohe` 前処理を推奨。
- `SVC` で `roc_auc` / `log_loss` を使う場合は `train.params.probability=true` が必須（確率推定のため計算コスト増）。
- `predict_proba` が無いモデルで確率必須の metric を指定すると train_model が明示的にエラーを返す。

## 1.4 Feature importance 出力（train_model）

- `feature_importances_` を持つモデルはその値を使用する（木系/GBDT）。
- `coef_` を持つモデルは絶対値を使用する（多クラスは平均絶対値）。
- 特徴量名が取得できない場合は `feature_0` 形式で補完する。


## 2. conf/group/model/*.yaml の推奨スキーマ

### 2.1 最小形（単一タスク種別）
```yaml
model_variant:
  name: linear_regression
  framework: sklearn
  class_path: sklearn.linear_model.LinearRegression
  params: {}
```

### 2.2 回帰/分類を同一名で切り替える（推奨）
`class_path` を dict にして `eval.task_type` で切替可能にします。
```yaml
model_variant:
  name: ridge
  framework: sklearn
  class_path:
    regression: sklearn.linear_model.Ridge
    classification: sklearn.linear_model.RidgeClassifier
  params:
    alpha: 1.0
    random_state: ${eval.seed}
```

---

## 3. optional dependency（導入方法）

GBDT や TabPFN は base 依存には入れず、**extras** で有効化します。

- GBDT系:
  - `pip install -e ".[models]"`  
- TabPFN:
  - `pip install -e ".[tabpfn]"`（または同等の extras）
  - `model_variant.params.auto_download`（default: false）
  - `auto_download=false` の場合は重みを事前取得しておく
  - `auto_download=true` の場合はライブラリの推奨手順で取得し、失敗時は out/manifest に記録される

> 運用上の重要ポリシー:
> - optional ライブラリが無い環境で 해당モデルを選んだ場合は **SKIP として明示**する  
>   例: `status=skipped` / `reason=missing_dependency` を記録し、導入手順を表示
> - 依存不足は `MissingOptionalDependencyError` を捕捉して SKIP 記録する（単体実行でも pipeline でも同じ）
> - pipeline v2 では missing deps は plan から除外される（`groups.train.skipped_missing_dependencies`）
> - “黙って他モデルに切り替える” は禁止（追跡性・再現性が壊れる）

---

## 4. 比較可能性（必ず守る）
- leaderboard は **task_type / split_hash / recipe_hash / primary_metric / direction** が一致しない run を混ぜない
- split は preprocess が固定（train が再生成しない）
