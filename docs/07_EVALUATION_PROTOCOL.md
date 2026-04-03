# 07 Evaluation Protocol

## 目的

このドキュメントは、train、leaderboard、pipeline で「何を同じ条件とみなして比較するか」を定義します。

## 比較可能性の基本条件

候補 run は次が一致しているときに比較可能とみなします。

- `task_type`
- `primary_metric`
- `direction`
- `processed_dataset_id` または `split_hash + recipe_hash`

推奨は次です。

- preprocess を 1 回作る
- その bundle を複数モデルに共有する
- すべての train が同じ split / recipe を使う

## split の扱い

split は preprocess の責務です。  
train_model は split を作り直しません。

そのため比較の中心は:

- `split_hash`
- `recipe_hash`

です。

## metric の扱い

### 回帰

典型的な primary metric:

- `r2`
- `rmse`
- `mae`
- `mse`

### 分類

典型的な primary metric:

- `accuracy`
- `roc_auc`
- `f1`

## leaderboard の比較ルール

### `leaderboard.require_comparable=true`

- 完全一致しない候補は除外
- `excluded_count` に反映

### `leaderboard.require_comparable=false`

- 比較対象に残す
- ただし warning や summary に明記

## ensemble の評価

ensemble も単体モデルと同じ評価軸を使います。  
selection metric は `ensemble.selection_metric` で明示し、既定は `eval.primary_metric` に揃えます。

## report で見るべき項目

- `primary_metric`
- `direction`
- `best_score`
- `split_hash`
- `recipe_hash`
- `processed_dataset_id`

## 実務上の注意

- split を train 側で変えると leaderboard の比較価値が下がる
- metric 名が同じでも task_type が違えば比較してはいけない
- pipeline で full set を回すときは preprocess を共通化する


