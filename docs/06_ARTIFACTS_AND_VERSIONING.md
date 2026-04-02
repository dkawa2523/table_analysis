# 06 Artifacts And Versioning

## Core artifacts
- `config_resolved.yaml`: resolved runtime config
- `out.json`: task result contract
- `manifest.json`: inputs, outputs, hashes, and versions

Solution code now calls the adapter family modules directly:
`platform_adapter_task.py`, `platform_adapter_artifacts.py`,
`platform_adapter_clearml_env.py`, `platform_adapter_model.py`,
and `platform_adapter_pipeline.py`.

## manifest.json
Typical fields:
- `schema_version`
- `created_at`
- `code_version`
- `platform_version`
- `process`
- `inputs`
- `outputs`
- `hashes`

## Hashes
- `config_hash`: resolved config identity
- `split_hash`: dataset split identity
- `recipe_hash`: preprocess recipe identity

## Versioning
- Solution schema version comes from `run.schema_version`
- Code and platform versions are resolved at runtime and recorded in the manifest and ClearML properties
