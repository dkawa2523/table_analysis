# 25 Alerting

## 目的

alerting は、運用上の重要イベントを見落としにくくするための補助層です。  
重い監視基盤を前提にせず、まずは file / stdout / webhook を揃えています。

## 何を alert にするか

- data quality fail
- drift warning / fail
- retrain trigger
- pipeline degraded result
- recommendation の重要変更

## 設定

```bash
run.alerts.enabled=true
```

主な設定:

- `run.alerts.sinks.file.enabled`
- `run.alerts.sinks.file.path`
- `run.alerts.sinks.stdout.enabled`
- `run.alerts.sinks.webhook.enabled`
- `run.alerts.sinks.webhook.url`

## sink の役割

### file

- JSONL で保存
- 最小構成で必ず残したい場合に使う

### stdout

- コンテナや CI で読みたいときに使う

### webhook

- 外部通知に流したいときに使う

## alert payload

各行は最低でも次を含みます。

- `timestamp`
- `kind`
- `severity`
- `title`
- `message`
- `usecase_id`
- `process`
- `context`

## ClearML との関係

ClearML 有効時は補助的に次も残します。

- tags
  - `alert:<kind>`
  - `severity:<level>`
- properties
  - `last_alert_kind`
  - `last_alert_severity`
  - `last_alert_title`
  - `last_alert_at`


