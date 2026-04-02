# ClearML試験段階ポリシー（固定しない／後で変えやすくする）

このドキュメントは **試験段階**（ローカルClearMLで検証 → 社内ClearMLで微調整）において、
運用ルールを“固定しない”代わりに、**後で設計判断しやすい**状態を作るための方針です。

## 試験段階で「やらない」こと

以下は本番運用で重要ですが、試験段階では **確定しません**。

- ClearMLサーバー側の権限設計（RBAC）を固める
- Queue名／Agent配置の最適化を固める
- プロジェクト命名則・タグ・Properties を単一仕様に固定する
- Template Task を組織標準として配布する（ただし作成手順は検討する）

## 試験段階で「やる」こと（必須）

### 1) 仕様候補を“並走”できるようにする

命名やタグの方針は複数案で試し、後で選ぶために、
コードではなく **yamlのポリシー差し替え**で切り替えられる状態を目指します。

例：
- `run.clearml.project_root`（上位階層の扱い。環境変数で上書き可）
- `ops/usecase_id_policy`（usecase_idの自動生成ルール）
- `ops/clearml_policy`（tags/propertiesの最小セット）

`ops/clearml_policy` で指定できるキー（例）：
- `run.clearml.policy.tags`
- `run.clearml.policy.properties`
- 互換のため `extra_tags` / `extra_properties` も受け付ける

```yaml
# conf/ops/clearml_policy/test_minimal.yaml
name: test_minimal
tags: []
properties: {}
```

dry-run での確認：
出力される tags/properties も `run.usecase_id_policy` の結果を反映する。
- `python -m tabular_analysis.ops.print_clearml_identity task=pipeline ops/clearml_policy=test_richer`
- `python -m tabular_analysis.ops.print_clearml_identity task=pipeline ops/clearml_policy=test_richer --now 20260101_120000`
- `python -m tabular_analysis.ops.print_clearml_identity task=pipeline ops/clearml_policy=test_richer --json`

### 2) 推薦（recommend）と採用の決定は分離

試験段階でも、
- recommendation は **自動**（leaderboardが出す）
- 採用は **推論時にユーザーが選択**（retrainでは選ばない）

を守ります。

### 3) ClearMLの情報量は増やしすぎない

- properties は **検索用の最小キー**だけ
- 詳細な説明・根拠・図表は **artifact（report/decision_summary）**へ

## 試験段階の成果物（必ず残す）

試験が進むほど、あとから「どの仕様が良かったか」が重要になります。
以下を docs/Issue 形式で残します。

- 命名則候補の比較（メリット/デメリット、UI見え方）
- タグ/Properties候補の比較（検索性、ノイズ）
- Template Task 運用案の比較（UI clone / pipeline clone / 手動実行）
- ローカルClearML→社内ClearML差分（修正ポイント）

## 試験段階の“事前リハーサル”

最低限、以下をローカルClearMLで1回通します。

1. localモード（ClearML無効）で pipeline を完走
2. loggingモード（ClearML有効・ローカル実行）で pipeline を完走
3. leaderboard / report / decision_summary / infer で選択を確認

手順は `docs/84_REHEARSAL_GUIDE.md` に記載します。
