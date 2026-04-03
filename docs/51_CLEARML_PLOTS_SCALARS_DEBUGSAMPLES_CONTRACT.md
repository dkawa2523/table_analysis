# 51 ClearML Plots Scalars And Debug Samples Contract

## 目的

artifact だけでなく、ClearML UI の `Scalars`、`Plots`、`Debug Samples` にも、運用上重要な情報を載せる契約です。

## 原則

- artifact は完全情報
- Scalars / Plots は即時理解用
- Debug Samples は例示と目視確認用

## Scalars の代表例

### train_model

- `metrics/<primary_metric>`
- `metrics/secondary/<metric>`

### leaderboard

- `leaderboard/best_score`

### pipeline

- `pipeline/num_models`
- `pipeline/num_succeeded`
- `pipeline/num_failed`

### infer

- `infer/latency_ms`

## Plots の代表例

### preprocess

- raw / processed summary
- missing / type / category summary

### train_model

- residual plot
- feature importance
- confusion matrix
- ROC curve

### leaderboard

- top-k bar chart
- metric comparison table

## Debug Samples の代表例

- preprocess: raw sample / processed sample
- train_model: y_true / y_pred sample
- infer: input / output sample

## 実装原則

- Plotly を優先
- PNG は artifact として残してもよいが、UI 表示は Plotly を優先
- summary table は Debug Samples や report table を使う


