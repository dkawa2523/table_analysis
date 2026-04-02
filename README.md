# ml-solution-tabular-analysis

Tabular classification/regression solution built on top of `ml-platform`.
The repository is organized around Hydra tasks, ClearML integration, and a
small set of operator-facing tools.

## Main entry points
- `python -m tabular_analysis.cli task=<task>`
- `python tools/clearml_templates/manage_templates.py --plan|--apply|--validate`
- `python tools/rehearsal/run_pipeline_v2.py --execution <local|logging|agent>`

## Key docs
- `docs/INDEX.md`
- `docs/65_DEV_GUIDE_DIRECTORY_MAP.md`
- `docs/67_REHEARSAL_COMMANDS.md`
- `docs/69_CLEARML_TROUBLESHOOTING.md`
- `docs/81_CLEARML_TEMPLATE_POLICY.md`
- `docs/84_REHEARSAL_GUIDE.md`

`docs/65_DEV_GUIDE_DIRECTORY_MAP.md` is the shortest code-navigation guide for engineers.
`docs/81_CLEARML_TEMPLATE_POLICY.md` is the canonical template operation contract.

## Verification
```bash
py -3 tools/tests/verify_all.py --quick
```

## Cleanup
```bash
py -3 tools/cleanup_repo.py --repo . --apply
```
