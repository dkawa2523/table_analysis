# 25_ALERTING (Operational Notifications)

## Goals
- Provide a reliable alert trail for operations (quality, drift, lifecycle events).
- Keep dependencies minimal: file logger by default, optional stdout/webhook.
- Surface alert state in ClearML tags/properties when ClearML is enabled.

## Configuration (Hydra)
Enable alerting explicitly:
```bash
run.alerts.enabled=true
```

Available settings (defaults in `conf/run/alerts/base.yaml`):
- `run.alerts.enabled` (default: false)
- `run.alerts.sinks.file.enabled` (default: true)
- `run.alerts.sinks.file.path` (default: null)
- `run.alerts.sinks.stdout.enabled` (default: false)
- `run.alerts.sinks.webhook.enabled` (default: false)
- `run.alerts.sinks.webhook.url` (default: null)
- `run.alerts.sinks.webhook.timeout_sec` (default: 5)

## Sinks
### File (default)
- JSONL output
- Default path: `<run.output_dir>/<stage>/alerts.jsonl`
- Override with `run.alerts.sinks.file.path`

### Stdout
- Prints one JSON line per alert when enabled

### Webhook
- HTTP POST with JSON body
- Only active when `url` is set and enabled

## Severity Guidance
- `info`: recommendation updates, routine operational events
- `warning`: drift warnings, model underperforming, retrain triggered
- `error`: data quality gate failures, drift fail threshold exceeded

## ClearML Integration
When ClearML is enabled, alerts add:
- tags: `alert:<kind>`, `severity:<level>`
- properties: `last_alert_kind`, `last_alert_severity`, `last_alert_title`, `last_alert_at`

## Log Format (JSONL)
Each line includes:
- `timestamp`, `kind`, `severity`, `title`, `message`
- Optional `usecase_id`, `process`, `stage`, `task_name`
- `context` (alert-specific payload)
