# ISSUE整理メモ

このファイルは、git 履歴から **後追いで再構成した issue 単位の整理メモ** です。  
GitHub Issue 番号そのものは commit message に明示されていなかったため、以下の `ISSUE-01` 以降は **commit message / 差分 / 変更ファイル** から論点ごとに束ね直したものです。

## 対象範囲

- 主対象: `8dec717` 〜 `2e55e0e`
- 転換点: `11dae2a Refine ClearML pipeline runtime and adapters`
- 調査方法:
  - `git log --reverse --format="%h %s"`
  - `git show --stat --summary <commit>`
  - 主要変更ファイルの現行コード読解

## 読み方

- `元の課題`: その issue 群が解決しようとしていた元の不具合・運用上の痛み
- `主なコミット`: 代表的な commit
- `主な変更箇所`: 特に意味の大きいファイル
- `変更による効果`: 運用・実装・UI のどこが改善されたか

---

## A. Adapter 整理以前〜並行して進んだ主要課題

### ISSUE-01 ClearML template 可視化と bootstrap/entrypoint の安定化

- 元の課題:
  - ClearML template の見え方、agent bootstrap、entrypoint override 解釈が不安定で、template 作成・実行環境立ち上げ・override 反映が壊れやすかった
  - queue binding や minimal requirements の考え方も揃っていなかった
- 主なコミット:
  - `8dec717 Fix ClearML uv bootstrap detection`
  - `911d530 Refine ClearML pipeline template visibility`
  - `22b64a9 Stabilize visible ClearML pipeline templates`
  - `2c17a5c Fix ClearML pipeline default queue binding`
  - `42a7d78 Apply minimal requirements to ClearML templates`
  - `54b3ea5 Minimize ClearML agent bootstrap requirements`
  - `1ad0620 Fix ClearML entrypoint slash override bootstrap`
  - `33cdc59 Restore ClearML uv bootstrap guard`
  - `c741a6a Normalize ClearML task overrides for entrypoint`
  - `6b2b395 Preserve config-group ClearML overrides`
  - `280c13f Stabilize ClearML loaded override merging`
- 主な変更箇所:
  - `tools/clearml_entrypoint.py`
  - `tools/clearml_templates/manage_templates.py`
  - `src/tabular_analysis/clearml/pipeline_templates.py`
  - `src/tabular_analysis/common/clearml_bootstrap.py`
  - `src/tabular_analysis/processes/pipeline.py`
  - `tools/clearml_agent/compose.yaml`
- 変更による効果:
  - template の publish と visible 化が安定した
  - entrypoint が slash override や config-group override を落としにくくなった
  - ClearML agent 側の必要依存が絞られ、bootstrap 失敗が減った
  - default queue の解釈が揃い、初期実行時の事故が減った

### ISSUE-02 Visible controller runtime と queue/controller 実行経路の安定化

- 元の課題:
  - visible pipeline controller の clone 実行、task id への attach、reseed、self-enqueue、plan-only、queue routing が複雑に絡み、controller run が不安定だった
  - step override への runtime ref 適用や recursive parameter parsing に抜けがあった
- 主なコミット:
  - `cd43c34 Attach ClearML pipeline controllers by task id`
  - `8fb2046 Fix ClearML controller runtime reseeding`
  - `33ae54b Restore pipeline template args on controller clone`
  - `512ee9a Run visible controllers in place on agent`
  - `b9de007 Start remote controllers without self-enqueue`
  - `54cc62f Preserve runtime refs in pipeline step overrides`
  - `64a67ad Apply runtime values to visible pipeline steps`
  - `52d0d5b Fix visible ClearML controller runtime`
  - `536b816 Fix ClearML services controller execution`
  - `e883c0d Preserve plan-only controller defaults`
  - `b23a9f5 Rename services queue to controller`
  - `501c8c0 Fix queue smoke routing and agent home isolation`
  - `b216919 Fix pipeline template smoke overrides`
  - `34fddf9 Fix recursive pipeline step parameter parsing`
- 主な変更箇所:
  - `src/tabular_analysis/processes/pipeline.py`
  - `src/tabular_analysis/processes/pipeline_support.py`
  - `src/tabular_analysis/ops/clearml_identity.py`
  - `tools/clearml_agent/compose.yaml`
  - `tools/tests/test_clearml_runtime_contracts.py`
- 変更による効果:
  - controller が seed clone から run controller へ正しく遷移しやすくなった
  - task id attach と runtime reseed が安定した
  - queue 名と役割が整理され、controller 用 queue の意味が明確化された
  - step parameter の run-time 参照が壊れにくくなった

### ISSUE-03 Pipeline lifecycle / override / report 契約の安定化

- 元の課題:
  - pipeline 実行の lifecycle と report 生成契約が揺れており、controller の状態、override の扱い、leaderboard bootstrap、report 再構築が一貫しなかった
- 主なコミット:
  - `3dfa69c Bootstrap leaderboard with planned model extras`
  - `8e2af52 Stabilize ClearML pipeline lifecycle contract`
  - `8a74931 Stabilize ClearML pipeline controller runs`
  - `4209b05 Align ClearML visible pipeline contracts`
  - `0944e5f Simplify ClearML pipeline override contract`
  - `b7c0178 Normalize ClearML template overrides`
  - `9778dd7 Harden ClearML override normalization`
  - `57d5f70 Attach remote pipeline controllers by task id`
  - `3b4ab6b Rebuild local pipeline reports from ClearML state`
- 主な変更箇所:
  - `src/tabular_analysis/processes/pipeline.py`
  - `src/tabular_analysis/reporting/pipeline_report.py`
  - `tools/rehearsal/run_pipeline_v2.py`
  - `src/tabular_analysis/platform_adapter_task_ops.py`
  - `tools/clearml_templates/manage_templates.py`
- 変更による効果:
  - lifecycle の実行結果と report 出力の対応が明確になった
  - override 契約が簡素化・正規化され、template 由来の揺れが減った
  - completed task から local report を再構築する補助手段が入り、運用後追いがしやすくなった
  - leaderboard bootstrap 時に planned model 情報を扱えるようになり、report の情報量が増えた

### ISSUE-04 Selection-driven fixed subset と pipeline profile の意味論整理

- 元の課題:
  - 固定 DAG の pipeline で profile、selection、ensemble defaults の優先順位が曖昧で、subset 実行時に「何が有効なのか」が読みにくかった
- 主なコミット:
  - `e60870d Add selection-driven fixed pipeline subsets`
  - `f98ac9b Honor pipeline profiles in selection runs`
  - `6746388 Restore ensemble profile method defaults`
  - `7fdea74 Use profile ensemble defaults in pipeline planner`
  - `f9512ad Prioritize pipeline profile run flags`
- 主な変更箇所:
  - `src/tabular_analysis/processes/pipeline.py`
  - `src/tabular_analysis/processes/pipeline_support.py`
  - `src/tabular_analysis/reporting/pipeline_report.py`
  - `conf/task/pipeline.yaml`
  - `conf/ensemble/base.yaml`
- 変更による効果:
  - profile 固定の意図を保ったまま subset だけ切り替えやすくなった
  - selection と profile defaults の優先順位が整理された
  - operator が見る report に requested/active/disabled の差分が出やすくなった

### ISSUE-05 Seed の NEW RUN ワークフロー、refresh、encoded key cleanup の完成

- 元の課題:
  - seed pipeline から NEW RUN した clone の扱い、code drift による refresh、usecase id の一意性、legacy encoded key の掃除、template lookup fallback などが未完成で、運用時のズレが残っていた
- 主なコミット:
  - `eba8c16 Stabilize template lock updates and ignore temp work dirs`
  - `d8448f5 Finalize ClearML seed NEW RUN workflow`
  - `e3044d0 Refresh pipeline seeds on code version drift`
  - `bd2ab10 Stabilize pipeline seed refresh behavior`
  - `644c61f Auto-generate unique UI pipeline usecase ids`
  - `6d772b5 Clean up legacy encoded ClearML parameter keys`
  - `35aea25 Fix pipeline UI parameter normalization`
  - `a69f7f6 Fix multi-encoded ClearML args cleanup`
  - `ab86670 Sync pipeline support runtime helpers`
  - `1aa6add Allow template lookup fallback via lock ids`
  - `5e9b802 Clean pipeline policy overrides in ClearML tasks`
  - `033e59c Guard pipeline UI clone normalization`
- 主な変更箇所:
  - `src/tabular_analysis/processes/pipeline.py`
  - `src/tabular_analysis/processes/pipeline_support.py`
  - `src/tabular_analysis/platform_adapter_task_ops.py`
  - `src/tabular_analysis/clearml/pipeline_templates.py`
  - `src/tabular_analysis/clearml/templates.py`
  - `tools/clearml_templates/manage_templates.py`
- 変更による効果:
  - NEW RUN clone の正規化が実運用レベルで安定した
  - code version drift 時の seed refresh 方針が明確化された
  - UI で発生していた encoded key 残骸の掃除が進み、見た目と source of truth のズレが減った
  - usecase id の自動採番により、operator が seed 既定値のまま起動しても run が衝突しにくくなった

### ISSUE-06 Docs / operator ガイド / agent セットアップの再整備

- 元の課題:
  - 実装と docs がずれやすく、operator 向け手順、agent セットアップ、内部設計の見取り図が不足していた
- 主なコミット:
  - `3deb1b5 Refine ClearML pipeline contracts and refresh docs`
  - `dede9ff Refine ClearML pipeline docs and agent setup`
- 主な変更箇所:
  - `docs/03_CLEARML_UI_CONTRACT.md`
  - `docs/17_OPERATOR_QUICKSTART.md`
  - `docs/52_CLEARML_PIPELINE_CONTROLLER_CONTRACT.md`
  - `docs/86_CLEARML_INTERNALS_FOR_DEVELOPERS.md`
  - `docs/87_CLEARML_PIPELINE_WORKFLOW_DETAILS.md`
  - `SETUP.md`
- 変更による効果:
  - operator 観点と developer 観点の docs が分離・充実した
  - 実装の正本がどこにあるか追いやすくなった
  - ClearML agent の立ち上げや troubleshooting の再現性が上がった

---

## B. Adapter 整理そのもの

### ISSUE-07 Adapter 層の責務分離と runtime/adapter の再編

- 元の課題:
  - adapter 層が肥大化し、query、policy、env、artifact、task context、pipeline integration が密結合だった
  - 変更時に「どこを触るべきか」が分かりにくく、影響範囲が大きかった
- 主なコミット:
  - `11dae2a Refine ClearML pipeline runtime and adapters`
- 主な変更箇所:
  - `src/tabular_analysis/platform_adapter_core.py`
  - `src/tabular_analysis/platform_adapter_common.py`
  - `src/tabular_analysis/platform_adapter_task_query.py`
  - `src/tabular_analysis/platform_adapter_clearml_policy.py`
  - `src/tabular_analysis/platform_adapter_task_context.py`
  - `src/tabular_analysis/clearml/live_cleanup.py`
- 変更内容:
  - 共通例外・tag/parameter 正規化・fileserver 補助を `platform_adapter_common.py` に分離
  - task 取得・tag/project/script/status query を `platform_adapter_task_query.py` に分離
  - policy / property / system tag / script override 周りを `platform_adapter_clearml_policy.py` に分離
  - runtime context / artifact upload / markdown report を `platform_adapter_task_context.py` に集約
  - legacy / deprecated pipeline cleanup を `clearml/live_cleanup.py` として独立
- 変更による効果:
  - adapter 修正の局所化が進み、後続 commit が細かい契約修正に集中できるようになった
  - ClearML integration の責務境界が見えやすくなり、保守性が大きく改善した

---

## C. Adapter 整理以後の修正・改良

### ISSUE-08 Post-refactor の HyperParameters / user properties / seed runtime 正規化

- 元の課題:
  - adapter 整理後も、seed / visible task / runtime task の HyperParameters と user properties の細部にノイズが残っていた
  - seed materialization 時の clean params、runtime defaults、property value 形式がまだ揺れていた
- 主なコミット:
  - `17f2dac Clean ClearML pipeline hyperparameter payloads`
  - `784a7c7 Clean visible ClearML pipeline hyperparameters`
  - `7f60c29 Handle seed materialization with clean pipeline params`
  - `632f3ae Restore seed runtime defaults from task identity`
  - `df99505 Normalize ClearML user property values in pipeline runtime`
  - `dfe838f Clean pipeline seed hyperparameters in ClearML`
  - `7c87229 Clean visible ClearML pipeline hyperparameters`
- 主な変更箇所:
  - `src/tabular_analysis/platform_adapter_task_ops.py`
  - `src/tabular_analysis/processes/pipeline.py`
  - `src/tabular_analysis/processes/pipeline_support.py`
  - `tools/clearml_entrypoint.py`
  - `tools/clearml_templates/manage_templates.py`
- 変更による効果:
  - visible pipeline と seed pipeline の HyperParameters 汚染がさらに減った
  - runtime defaults を task identity から戻す経路が入り、NEW RUN 後の意味論が安定した
  - user properties の値形式が揃い、比較や downstream 利用がしやすくなった

### ISSUE-09 NEW RUN HyperParameters と UI contract の再定義

- 元の課題:
  - operator が UI で何を編集すべきか、bootstrap key と editable key の境界が曖昧で、NEW RUN clone の HyperParameters が意図通りに見えないケースがあった
- 主なコミット:
  - `b6811ac Fix ClearML pipeline NEW RUN hyperparameters`
  - `017ca88 Simplify ClearML pipeline queue and UI contracts`
- 主な変更箇所:
  - `src/tabular_analysis/clearml/pipeline_ui_contract.py`
  - `src/tabular_analysis/processes/pipeline.py`
  - `src/tabular_analysis/processes/pipeline_support.py`
  - `docs/03_CLEARML_UI_CONTRACT.md`
  - `docs/61_CLEARML_HPARAMS_SECTIONS.md`
  - `docs/17_OPERATOR_QUICKSTART.md`
- 変更による効果:
  - profile ごとの UI whitelist が明確化された
  - `Hyperparameters` と `Configuration > OperatorInputs` の役割分担が整理された
  - NEW RUN 時の編集体験が一貫し、operator の誤操作リスクが下がった

### ISSUE-10 Seed materialization の最終安定化と branch-mode drift 判定の緩和

- 元の課題:
  - seed materialization の最終段で template lock と branch-mode drift 判定が厳しすぎる、または refresh と衝突する可能性が残っていた
- 主なコミット:
  - `7e41477 Stabilize ClearML seed materialization`
  - `0396cca Update ClearML template lock`
  - `2e55e0e Relax branch-mode seed code drift checks`
- 主な変更箇所:
  - `src/tabular_analysis/clearml/pipeline_templates.py`
  - `tools/clearml_templates/manage_templates.py`
  - `conf/clearml/templates.lock.yaml`
- 変更による効果:
  - template lock の更新と seed materialization の整合が取りやすくなった
  - branch-mode の code drift 判定が過剰に seed refresh を要求しにくくなった
  - seed 管理の運用負荷が下がった

---

## D. 推論まわりで確認できた修正

### ISSUE-11 推論処理の責務分離と batch/optimize/drift 補助ロジックの整理

- 元の課題:
  - `infer.py` に batch 実行、chunk 読み込み、optimize 設定、dry-run、drift 用前処理、しきい値や calibration の解決などが集まりすぎており、pipeline 側の修正と一緒に保守しづらかった
  - 推論結果の解釈に必要な train profile、class label、threshold、calibration 情報の取り回しも task 本体に埋まりがちで、責務境界が見えにくかった
- 主なコミット:
  - `8e2af52 Stabilize ClearML pipeline lifecycle contract`
  - `d8448f5 Finalize ClearML seed NEW RUN workflow`
  - `dede9ff Refine ClearML pipeline docs and agent setup`
  - `11dae2a Refine ClearML pipeline runtime and adapters`
- 主な変更箇所:
  - `src/tabular_analysis/processes/infer.py`
  - `src/tabular_analysis/processes/infer_support.py`
  - `src/tabular_analysis/reporting/pipeline_report.py`
- 変更内容:
  - chunk 読み込み、batch input 解釈、optimize search space/hparams 解決、preprocess 列解決を `infer_support.py` に移した
  - dry-run、train profile 読み込み、drift 用 frame 正規化、threshold/calibration/class label 解決を helper 化した
  - pipeline report 側も live state 再構築の流れに合わせて、推論 task から拾う情報の扱いを整理した
- 変更による効果:
  - `infer.py` が orchestration 中心になり、推論ロジックの見通しが良くなった
  - batch 推論、optimize、drift 監視の補助処理が再利用しやすくなった
  - 推論結果に付随する metadata の解釈が揃い、report や artifact 表示の一貫性が上がった

### ISSUE-12 推論 child queue と execution policy 契約の明確化

- 元の課題:
  - 推論 child task の enqueue 先が `run.clearml.queue_name` 前提に寄っており、pipeline 側で queue 契約を整理したあとに infer batch child だけ古い前提を引きずる余地があった
  - operator から見ると、controller / default / infer の queue 使い分けが UI 契約と完全には揃っていなかった
- 主なコミット:
  - `017ca88 Simplify ClearML pipeline queue and UI contracts`
- 主な変更箇所:
  - `src/tabular_analysis/processes/infer.py`
  - `docs/03_CLEARML_UI_CONTRACT.md`
  - `docs/61_CLEARML_HPARAMS_SECTIONS.md`
- 変更内容:
  - infer child task の queue 解決を `exec_policy.queues.infer` 優先、未設定なら `exec_policy.queues.default` fallback に寄せた
  - queue 未設定時のエラーメッセージも infer 専用 queue 契約に合わせて明示化した
- 変更による効果:
  - pipeline 全体の execution policy と infer child 実行の整合が取れた
  - batch infer 実行時に queue 設定の意味を operator が理解しやすくなった
  - `run.clearml.queue_name` 依存の残骸が減り、UI 契約の統一感が増した

---

## 全体所感

- 大きな流れは `bootstrap/visible化` → `controller 実行安定化` → `override / report / profile 契約整理` → `seed NEW RUN 完成` → `adapter 分解` → `post-refactor cleanup` です。
- `11dae2a` の adapter 分解は単独の改善ではなく、その後の修正を局所化するための基盤整備として効いています。
- docs と `tools/tests/test_clearml_runtime_contracts.py` が毎段階で更新されているため、実装だけでなく契約そのものを徐々に固めていった履歴だと読めます。

## 追記候補

- 各 issue ごとに「代表 diff の抜粋」を入れる
- issue ごとに「影響した operator 導線」を追加する
- issue ごとに「残課題 / 未整理点」を追加する


