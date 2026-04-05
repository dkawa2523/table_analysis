# 13 Platform Integration

## 目的

このファイルは、solution repo が `ml_platform` とどう接続しているかを説明します。  
platform 側の詳細実装ではなく、「solution からどこを経由して呼ぶか」を把握するための文書です。

## 原則

- process から platform API を広く直接呼ばない
- 接続面は adapter family に寄せる
- ClearML / script / artifact / model / pipeline ごとに責務を分ける

## Canonical modules

- `src/tabular_analysis/platform_adapter_core.py`
  - script、version、project、template、queue などの共通 helper
- `src/tabular_analysis/platform_adapter_task.py`
  - task lifecycle、tags、properties、hparams 接続
- `src/tabular_analysis/platform_adapter_task_ops.py`
  - task 操作、parameter 設定、tag 更新
- `src/tabular_analysis/platform_adapter_artifacts.py`
  - artifact / manifest / upload helper
- `src/tabular_analysis/platform_adapter_model.py`
  - model registry / model reference 解決
- `src/tabular_analysis/platform_adapter_pipeline.py`
  - ClearML pipeline controller helper
- `src/tabular_analysis/platform_adapter_clearml_env.py`
  - ClearML runtime / env helper

## この分割の理由

- task の本体が ClearML 詳細で汚れない
- testing 時に責務を追いやすい
- ClearML 契約変更を adapter family に集約できる

## まだ solution 側に残るもの

- tabular 固有の dataset contract
- model variant と optional dependency の解決
- seed pipeline の運用ポリシー

## 関連ドキュメント

- [01_POLYREPO_INTENT.md](01_POLYREPO_INTENT.md)
- [54_CLEARML_MINIMALITY_GUIDE.md](54_CLEARML_MINIMALITY_GUIDE.md)
- [65_DEV_GUIDE_DIRECTORY_MAP.md](65_DEV_GUIDE_DIRECTORY_MAP.md)


