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

## 検証

```bash
python tools/tests/verify_all.py --quick
```


