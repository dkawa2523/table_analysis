# 26 Reproducible Runtime

## 目的

この repo では task-time install を維持しつつ、実行ごとの差分を追えるようにしています。  
その中心は `uv.lock`、task-specific extras、env snapshot です。

## 基本方針

- task ごとに独立実行
- venv は再利用しない
- `uv` cache は共有する
- optional dependency は task / model 単位で最小化する

## ローカルの基準

```bash
uv sync --frozen
```

optional dependency 例:

```bash
uv sync --frozen --extra lightgbm
uv sync --frozen --extra xgboost
uv sync --frozen --extra catboost
```

## ClearML task-time bootstrap

ClearML 側では `tools/clearml_entrypoint.py` が task ごとの extra を解決して `uv sync` を行います。

例:

- `pipeline`, `preprocess`, `leaderboard`
  - base only
- `lgbm`
  - `lightgbm`
- `xgboost`
  - `xgboost`
- `catboost`
  - `catboost`

## Agent 側の前提

- `uv` binary は agent image に入れておく
- `UV_CACHE_DIR=/root/.clearml/uv-cache`
- `/root/.clearml` は volume 共有
- container は `init: true`

## snapshot artifact

各 task は runtime snapshot を残します。

- `env.json`
- `pip_freeze.txt`

これらは local file にも ClearML artifact にも残ります。

## lockfile 更新

依存を変えたとき:

```bash
uv lock
```

## pip fallback

pip-only 環境では次を使います。

```bash
pip install -r requirements/base.txt
pip install -e .
```


