# Serving (Optional FastAPI)

This repository includes a lightweight serving skeleton for integrating a trained
`model_bundle.joblib` into an API. FastAPI/uvicorn are optional extras and are
not installed by default.

## Install optional dependencies

```bash
pip install -e ".[api]"
```

## Run with uvicorn

```bash
export TABULAR_MODEL_BUNDLE=/path/to/model_bundle.joblib
uvicorn tabular_analysis.serve.app:app --reload
```

## Programmatic usage

```python
from tabular_analysis.serve import create_app

app = create_app("/path/to/model_bundle.joblib", strict_schema=False)
```

## Endpoints

`GET /health`
- Returns `status`, the resolved model reference, and `error` when the bundle cannot be loaded.

`POST /predict`
- Accepts a single record (object) or a list of records, or wrapped as
  `{"records": [...]}` / `{"inputs": [...]}`.
- Returns `predictions` and `schema_validation`.

Example:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"records":[{"num1":1.2,"num2":3.4,"cat":"a"}]}'
```

## Schema validation

- `strict_schema=False` (default): coerce types and continue with warnings.
- `strict_schema=True`: return HTTP 422 if schema issues are found.

