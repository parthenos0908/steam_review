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
学習時はcombied（前処理をしてタイトルと本文をつなげた文）を入力として用いる

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
    "review": "Great game. Solid content. Playing since 2012. Recommended for trucking enthusiast, driving thousand kilometers at 80 kph.\nValue 10/10.",
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
学習時はreview_lem（前処理したレビュー文）を入力として用いる
色々情報が載っているがreview_lemとlabel以外特に使わない

## `code/classify/classifier.py`の使い方
`steam_review/code/classify`で以下を実行
`$ python classifier.py [教師データ] [テストデータ] [mode]`
