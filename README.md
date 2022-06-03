# steam_review

## ゲームID
- 227300：Euro truck simulator 2
- 255710：Cities: Skylines
- 427520：factorio（実験では未使用）

## dataの各ファイル説明（xxxは上記ゲームIDに置き換えてください）
- `xxx_forum.json`：`code/scraping/get_forum_xxx.json`の実行結果
- `xxx_forum_cleaned.json`：`xxx_forum.json`に`code/preprocessing/preprocessing.py`した実行結果
- `xxx_review.json`：`code/scraping/get_review.json`の実行結果（xxxは実行時指定）
- `xxx_review_cleaned.json`：`xxx_review.json`に`code/preprocessing/preprocessing.py`した実行結果
- `xxx_review_cleaned_out.json`：`xxx_review_cleaned.json`を目視でラベル付けした結果．  
ラベルの対応は，0：バグ報告，1：機能要求，2：その他，3：英語以外，-1：未分類

それ以外は`code/labeling/labeling.py`でラベリングする際に，jsonがデカすぎたので分割したファイルです．

### `xxx_forum_cleaned.json`の形式
```json
{
    "combined": "rigid truck who else think that 's rigid truck should be part of ets at to think would be cool correct spelling",
    "comment": "Who else thinks that's rigid trucks should be a part of ets and ats to. I think would be cool.\n\nCorrected spelling",
    "comment_lem": "who else think that 's rigid truck should be part of ets at to think would be cool correct spelling",
    "id": 14,
    "label": 2,
    "num_words": 22,
    "title": "Rigid trucks",
    "title_lem": "rigid truck",
    "url": "https://forum.scssoft.com/viewtopic.php?f=41&t=304816&sid=756637e85960522bd70695712dbd8353"
},
```
学習時はcombied（前処理をしてタイトルと本文をつなげた文）を入力として用いる．

### `xxx_review_cleaned_out.json`の形式
```json
{
    "author": {
        "last_played": 1606089143,
        "num_games_owned": 45,
        "num_reviews": 6,
        "playtime_at_review": 31975,
        "playtime_forever": 31975,
        "playtime_last_two_weeks": 0,
        "steamid": "76561198164237334"
    },
    "comment_count": 0,
    "id": 0,
    "label": -1,
    "language": "english",
    "num_words": 19,
    "received_for_free": false,
    "recommendationid": "101210709",
    "review": "Great game. Solid content. Playing since 2012. Recommended for trucking enth，usiast, driving thousand kilometers at 80 kph.\nValue 10/10.",
    "review_lem": "great game solid content play since 2012 recommend for truck enthusiast drive thousand kilometer at 80 kph value 10/10",
    "steam_purchase": false,
    "timestamp_created": 1634492456,
    "timestamp_updated": 1634492456,
    "voted_up": true,
    "votes_funny": 0,
    "votes_up": 0,
    "weighted_vote_score": 0,
    "written_during_early_access": false
},
```
学習時はreview_lem（前処理したレビュー文）を入力として用いる．  
色々情報が載っているがreview_lemとlabel以外特に使わない

## `code/classify/classifier.py`の使い方
以下を実行（相対パスでファイルを指定しているので`code/classify`の中で実行してください）  
`$ python classifier.py [ID1] [ID2] [MODE]`  
例：`$ python classifier.py 255710 255710 r`
- ID1：ゲームのID
- ID2：ゲームのID
- MODE：r(review), f(forum), c(cross) から選択
    - r：ID1の**review**を教師データとしてID1の**review**を分類する（ID2関係なし）
    - f：ID1の**forum**を教師データとしてID1の**review**を分類する（ID2関係なし）
    - c：ID2の**forum**を教師データとしてID1の**review**を分類する（ID1=ID2ならMODE=fと同じ）

実行結果は`code/classify/result`に保存される．**既に存在するファイルは上書きされるので，上書きされたくない場合はresultフォルダを別名に変更してください**

`code/classify/classifier.bat`で今回実験を行った6パターン（R_Cities ⇒ R_Cities, R_Cities ⇒ F_Cities...）を一度に実行できます

### パラメータの設定
`steam_review/code/classify/setting`は設定ファイルです．学習/分類時の各パラメータを変更できます
```python
# DLパラメータ
num_classes = 3
max_length = 128  # 256はメモリ不足
batch_size = 32
max_epochs = 10
hold_out_rate = 0.7  # 訓練データとテストデータの比率

# 設定
is_learn = True  # 新しく学習する(True) or 既存の学習結果使う(False)
is_del_less_words = False  # 1単語以下をテストデータから除外
is_less_train_data = False  # フォーラムの教師データ数をレビューに合わせる
is_earlystop = True  # 早期終了を行う
```
- `is_learn = True`の場合，学習と分類を行います．resultフォルダがない場合は新たに作成されます．学習結果に再現性はありません．基本こちらを使います．  
- `is_learn = False`の場合，学習は行わず分類のみ行います．resultフォルダ内の`xxx/xxx_MODE_model/checkpoint`を参照して，以前`is_learn = True`で実行した際に保存したモデルデータを参照して分類を行います．モデルが同一であれば分類結果は再現性があります．
- データはランダムにシャッフルされたのち訓練データとテストデータに分けられますが，seedを固定しているので再現性があります．

### 前処理を行わないで実験したい場合
59，71行目`train_texts.append(train_d['review_lem'])`→`train_texts.append(train_d['review'])`に変更  
88行目`train_texts.append(forum_d['combined'])`→`train_texts.append(forum_d['title'] + forum_d['comment'])`に変更  
で上手くいくと思います
https://github.com/parthenos0908/steam_review/blob/50f0fbe59b1a3bc8221fe54f61de2a48542b369e/code/classify/classifier.py#L54-L90


## `code/preprocessing/preprocessing.py`の使い方
- スクレイピングした直後のreview,forumに前処理（小文字化，ストップワード除去，lemmatize）を行います
- コード内の変数`INPUT_FILENAME`と`OUTPUT_FILENAME`と`MODE`を適宜変更して実行してください．コマンドライン引数は不要です．
- `code/preprocessing`に`INPUT_FILENAME`ファイルを置いて実行することで，同フォルダに`OUTPUT_FILENAME`ファイルが生成されます．
- `MODE`はF(forum)かR(review)を設定してください．

## link
- [実験メモ](https://hackmd.io/ufssII94QwC2EuZRT1PcUw)
- [目視ラベリングの基準決め](https://hackmd.io/O0oe4PYNQWaVdcLDaL35Pg)
