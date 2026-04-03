# 18 Contributing

## 目的

このガイドは、日常開発で迷わないための最小ルールをまとめたものです。

## セットアップ

### uv を使う標準手順

```bash
uv sync --frozen
```

optional dependency:

```bash
uv sync --frozen --extra lightgbm --extra xgboost --extra catboost
uv sync --frozen --extra api
```

### pip fallback

```bash
pip install -r requirements/base.txt
pip install -e .
```

## 変更時の原則

- process file は orchestration に寄せる
- pure helper は `common/` や `registry/` に寄せる
- ClearML API 呼び出しは adapter / clearml family に寄せる
- config を変えたら `conf/` と `docs/` を同時に更新する

## PR の最低条件

- 目的と影響範囲を書く
- 検証コマンドを書く
- operator 向け挙動が変わるなら docs を更新する
- generated files を不要に commit しない

## よく使う確認

```bash
python tools/tests/check_docs_paths.py --repo .
python tools/tests/test_template_specs.py
python tools/tests/test_clearml_runtime_contracts.py
python tools/tests/verify_all.py --quick
```

## docs を更新すべきケース

- ClearML の見え方が変わる
- queue / agent / template 運用が変わる
- 新しい model variant を増やす
- report / leaderboard の契約が変わる

## コミット前に見ておきたい場所

- `README.md`
- `docs/INDEX.md`
- `docs/65_DEV_GUIDE_DIRECTORY_MAP.md`
- `docs/81_CLEARML_TEMPLATE_POLICY.md`
- `docs/82_CLEARML_PROJECT_LAYOUT.md`


