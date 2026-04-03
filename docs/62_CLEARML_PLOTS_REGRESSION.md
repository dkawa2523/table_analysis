# 62 ClearML Plots Regression

## 目的

回帰タスクで、ClearML UI に何を出すべきかを整理したガイドです。

## Scalars

代表的な regression scalar:

- `metrics/r2`
- `metrics/mse`
- `metrics/rmse`
- `metrics/mae`

## preprocess で出したいもの

- raw sample summary
- processed sample summary
- missing / type / category summary

## train_model で出したいもの

- metrics table
- y_true vs y_pred
- residual plot
- feature importance

## leaderboard で出したいもの

- model ranking table
- top-k bar chart
- score comparison

## infer で出したいもの

- input / output table
- batch summary

## 実装の考え方

- Plotly 優先
- artifact は完全情報
- UI は理解用に絞る


