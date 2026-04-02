# 27_DATA_QUALITY (Data Quality Gate)

## Goals
- Detect common data defects early (missing, duplicates, constant columns, mixed types).
- Surface leakage risks (exact match, near match, high correlation, deterministic mappings).
- Flag ID-like columns (near-unique) and high-cardinality categorical columns.
- Keep a lightweight trail in artifacts and ClearML properties.

## When It Runs
- `dataset_register`, `preprocess`, and `infer`.
- Controlled by `data.quality.mode`:
  - `warn` (default): log issues, do not stop the run.
  - `fail`: stop the run when any fail-level issue is detected.
  - `off`: skip checks (still writes a minimal report).

## Outputs
Artifacts in each task output directory:
- `data_quality.json`
- `data_quality.md`

Key fields in `data_quality.json`:
- `quality_status`: pass | warn | fail
- `quality_issue_count`
- `quality_issues`: per-rule issues with severity
- `leak_suspects`, `id_like_columns`, `high_cardinality_columns`, `mixed_type_columns`, `name_suspects`

## ClearML Integration (Minimal)
Properties:
- `quality_status`
- `quality_issue_count`

Artifacts:
- `data_quality.json`
- `data_quality.md`

## Configuration
Defaults live in `conf/data/quality/base.yaml`.
Override with Hydra, for example:
```bash
python -m tabular_analysis.cli task=preprocess \
  data.quality.mode=fail \
  data.quality.thresholds.high_cardinality_ratio_warn=0.9
```

Key settings:
- `data.quality.mode`: warn | fail | off
- `data.quality.max_rows_scan`: sample cap for large datasets
- `data.quality.thresholds.*`: rule thresholds (missing/duplicates/type drift/leak/id/high-card)
- `data.quality.name_patterns.*`: suspicious name patterns

## Notes
- Checks operate on a sample when `rows_total > max_rows_scan`.
- Fail-level issues default to severe leaks/defects; adjust thresholds for stricter gates.
- Mixed-type columns are detected on sampled values from object-typed columns.
