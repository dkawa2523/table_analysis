# 27 Data Quality

## 目的

data quality gate は、欠損、重複、constant、mixed type、leakage、ID-like column などの典型的な問題を早期に見つけるための仕組みです。

## いつ動くか

- `dataset_register`
- `preprocess`
- `infer`

## mode

- `warn`
  - 記録するが止めない
- `fail`
  - fail-level issue で停止する
- `off`
  - 最小 report のみ

## 主な出力

- `data_quality.json`
- `data_quality.md`

主な項目:

- `quality_status`
- `quality_issue_count`
- `quality_issues`
- `leak_suspects`
- `id_like_columns`
- `high_cardinality_columns`
- `mixed_type_columns`

## ClearML への反映

properties:

- `quality_status`
- `quality_issue_count`

## よく見る設定

- `data.quality.max_rows_scan`
- `data.quality.thresholds.*`
- `data.quality.name_patterns.*`

例:

```bash
python -m tabular_analysis.cli task=preprocess \
  data.quality.mode=fail \
  data.quality.thresholds.high_cardinality_ratio_warn=0.9
```

## 運用上の考え方

- 大規模データでは sample ベースでもよい
- leak や severe defect は fail に寄せる
- warn でも report には明示的に残す


