# 85 Exception Handling Policy

## 目的

例外処理を曖昧にせず、recoverable error と fatal error を区別するためのポリシーです。

## 原則

1. broad except を避ける
2. recoverable な例外だけを狭く捕まえる
3. fallback は silent にしない
4. optional dependency の欠如は明示的に report / skip する

## 典型例

### 許容されるもの

- `ImportError`
- `ModuleNotFoundError`
- `ValueError`
- `FileNotFoundError`

### 避けるもの

- bare `except:`
- 何でも `except Exception` で握りつぶす

## 運用上の扱い

- task が継続不能なら fail-fast
- task が skip 可能なら reason を残して skip
- ClearML best-effort 部分は warning に寄せるが、契約違反は隠さない

## pipeline_controller で特に重要な fail-fast

current pipeline 運用では、次を silent fallback にしません。

- `data.raw_dataset_id` が empty
- `data.raw_dataset_id` が seed placeholder `REPLACE_WITH_EXISTING_RAW_DATASET_ID` のまま
- fixed profile に存在しない `pipeline.selection.*` / `ensemble.selection.*`
- `run.usecase_id_policy=explicit` なのに `run.usecase_id` が空

これらは operator mistake を hidden success にしないため、早い段階で失敗させます。

## ClearML best-effort と契約違反の違い

### best-effort 側

- UI 上の過去 task に `%2E` を含む historical key が残る
- fileserver artifact の補助取得が失敗する
- optional plot / debug sample の欠落

### fail-fast 側

- seed placeholder のまま actual run を開始
- seed clone の metadata が run shape に正規化できない
- required queue / required seed / required contract tag が欠落

この区別が、operator 向けの分かりやすさと developer 向けの保守性の両方で重要です。

## 検証

```bash
python tools/tests/verify_all.py --quick
```


