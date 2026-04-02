# 43 ClearML UI Layout Examples

This document provides lightweight examples for checking whether the current UI
layout is still readable.

## Example project grouping
```text
<ROOT>/<solution_root>/<usecase_id>/01_Datasets
<ROOT>/<solution_root>/<usecase_id>/02_Preprocess
<ROOT>/<solution_root>/<usecase_id>/03_TrainModels
<ROOT>/<solution_root>/<usecase_id>/00_Pipelines
<ROOT>/<solution_root>/<usecase_id>/05_Infer
```

## What to compare
- Process groups are easy to scan.
- Pipeline tasks are visually separate from train tasks.
- Infer child tasks are isolated from the main infer group when present.
- Dataset and preprocess tasks are not mixed into training groups.

## Recording differences
Capture the final decision in the PR description, an operations note, or your
team's issue tracker.
