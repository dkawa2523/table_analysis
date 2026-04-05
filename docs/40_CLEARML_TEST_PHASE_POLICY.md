# 40 ClearML Test Phase Policy

## 目的

ClearML まわりの変更を、いきなり本番運用へ載せず、段階的に確認するためのテスト方針です。

## 基本方針

変更は次の順で確認します。

1. local
2. logging
3. template / seed の apply / validate
4. pipeline_controller
5. operator UI 確認

## Phase 1: local

目的:

- task ロジックそのものの確認
- artifact / out.json / manifest の確認

代表コマンド:

```bash
python tools/tests/smoke_local.py --until pipeline
```

## Phase 2: logging

目的:

- ClearML task と artifact が正しく見えるか
- HyperParameters / tags / properties の確認

代表コマンド:

```bash
python tools/rehearsal/run_pipeline_v2.py \
  --execution logging \
  --task-type regression \
  --project-root LOCAL
```

## Phase 3: template / seed refresh

目的:

- template / seed metadata drift の検出
- seed pipeline の存在確認

代表コマンド:

```bash
python tools/clearml_templates/manage_templates.py --apply --project-root LOCAL
python tools/clearml_templates/manage_templates.py --validate --project-root LOCAL
```

## Phase 4: pipeline_controller

目的:

- controller queue / child queue の動作確認
- seed pipeline の `NEW RUN` 実行確認

確認点:

- controller は `controller`
- heavy model は `heavy-model`
- light child は `default`

## Phase 5: operator UI

目的:

- `.pipelines/<profile>` project で seed pipeline card が見える
- `NEW RUN` / rerun ができる
- project tree から child task を辿れる

## 推奨 gate

- docs path check
- template spec test
- clearml runtime contract test
- quick verify
- rehearsal UI verify

## 非推奨

- hidden fallback に頼る
- template apply を省略して validate だけ回す
- operator 運用前に logging を飛ばす

