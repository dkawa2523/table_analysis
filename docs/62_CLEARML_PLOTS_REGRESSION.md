# ClearML PLOTS/SCALARS 可視化 契約（回帰中心） v1

## 共通指標（Scalars）
- `metrics/r2`
- `metrics/mse`
- `metrics/rmse`
- `metrics/mae`

## dataset_register / preprocess（PLOTS）
- データ概要テーブル（head sample）: Plotly Table
- 欠損率（bar）: Plotly
- 数値特徴の分布（上位K列）: Plotly
- カテゴリ特徴の top-k（上位K列）: Plotly
- 目的変数分布（回帰: histogram / 分類: top-k）: Plotly
- preprocess は raw vs processed のサマリー表（rows/features/types/missing）と欠損率比較 bar を追加

## train_model（PLOTS）
- 指標テーブル（R2/MSE/RMSE/MAE）: Plotly Table
- y_true vs y_pred 散布図 + y=x ライン（R2表示）: Plotly
- residual plot（任意）: Plotly
- feature importance（取得できるモデルのみ）: Plotly bar
- CV が有効な場合: fold別スコアのbox/violin（任意）
- series name（推奨）: `metrics_table`, `true_vs_pred`, `residuals`, `feature_importance`

## leaderboard（PLOTS）
- モデル/前処理/各指標（R2/RMSE/MAE/MSE）/composite_score のテーブル: Plotly Table（推奨行をハイライト）
- 総合スコア（composite_score）上位Kのbar: Plotly
- パレート散布図（例: R2 vs RMSE）: Plotly（任意）
- scoring は `conf/leaderboard/scoring.yaml` の weights/normalization で制御

## infer（PLOTS）
- **入力→出力のテーブル**（single/batch/optimize 共通）: Plotly Table（series: `input_output_table`, columns: `in.*`/`out.*`）
- batch: 予測値分布（hist）+ 上位/下位のサンプルテーブル（任意）
- optimize: Optunaグラフ（history, parallel coords, importance, response surface）
- optimize: 上位K trial の入力→出力テーブル（series: `input_output_table`）
