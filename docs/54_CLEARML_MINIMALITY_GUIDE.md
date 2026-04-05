# 54 ClearML Minimality Guide

## 目的

ClearML 対応コードが散らかるのを防ぐための設計ガイドです。

## 原則

1. ClearML API 呼び出しは adapter family か `src/tabular_analysis/clearml/` に寄せる
2. process file は orchestration と domain logic に集中させる
3. compatibility alias より canonical contract を優先する
4. hidden fallback より fail-fast を優先する

## どこに何を書くか

### process file

- task の流れ
- domain の判断

### adapter family

- Task / artifact / project / queue / script / model / env の接続

### clearml package

- template
- hparams
- seed pipeline
- UI logger

## やってはいけないこと

- 同じ tag 生成を複数ファイルに書く
- process 本体から ClearML SDK を直接散発的に呼ぶ
- template lookup に曖昧な fallback を足す


