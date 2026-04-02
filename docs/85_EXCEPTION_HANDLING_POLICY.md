# Exception Handling Policy

## Objective
- 読解性と障害切り分けを維持するため、`except Exception` と bare `except:` を禁止する。
- recoverable な失敗だけを明示的にハンドルし、想定外は上位へ伝播させる。

## Rules
1. `except Exception` / `except:` は使用しない。
2. fallback 用の `except` は、対象を `ImportError` / `ModuleNotFoundError` や `ValueError` などに限定する。
3. 複数箇所で同じ分類を使う場合は、モジュール先頭で `*_RECOVERABLE_ERRORS` としてまとめる。
4. 例外を握りつぶす場合は、機能低下が許容されるケース（optional dependency, placeholder rendering, best-effort logging）に限定する。
5. I/O・外部API・ClearML連携の失敗は、`PlatformAdapterError` などの境界例外に正規化して再送出する。

## Verification
- ローカル/CI 共通で `tools/tests/check_broad_excepts.py` を実行し、`src/tabular_analysis` 配下に broad except がないことを保証する。
- `tools/tests/verify_all.py --quick` では `broad_except_check` を必須ステップとして実行する。

## Notes
- 方針は挙動変更ではなく可観測性向上が目的。既存の fallback 動作は維持し、例外型だけを狭める。
- 2026-03-01 時点で `src/tabular_analysis` の broad except は 0 件を維持。
- 2026-03-01 時点で `ml_platform` 側 `src/ml_platform` も broad except は 0 件。
- adapter 分割後も例外正規化の境界は `PlatformAdapterError` に統一する（shim廃止予定: 2026-07-01 以降）。
