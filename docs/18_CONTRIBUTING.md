# 18 Contributing

## Scope
This guide documents the minimum workflow for day-to-day development and PRs.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/base.txt
pip install -e .
```

`uv` alternative:
```bash
uv sync --frozen
```

Optional model extras:
```bash
uv sync --extra models
uv sync --extra tabpfn
```

## Pull Request Checklist
- Explain the intent, scope, and verification status.
- Update docs when changing UI contracts, config behavior, or operator workflows.
- Do not commit generated files under `work/`, `artifacts/`, or local ClearML output directories.

## Verification
```bash
python tools/tests/verify_all.py --quick
```

Use the full suite when preparing a release or a larger refactor:

```bash
python tools/tests/verify_all.py --full
```

## Repository Constraints
- Platform API dependencies stay behind the adapter family modules under `src/tabular_analysis/`.
- Keep `conf/` and `docs/` in sync when behavior changes.
- Update `docs/03_CLEARML_UI_CONTRACT.md` when changing visible ClearML behavior.
