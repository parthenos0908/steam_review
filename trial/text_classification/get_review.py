import requests
import json
from os import path

game_id = 227300
url = "https://store.steampowered.com/appreviews/" + str(game_id) + "?json=1"
params = {
    "filter": "all",
    "language" : "english",
    "day_range" : 9223372036854775807, # n日前までのreviewを取得(bigintの最大値を入力)
    "review_type" : "all",
    "purchase_type" : "all",
    "num_per_page" : 100,
    "start_offset": 0,
    "cursor" : "*"
}
# https://partner.steamgames.com/doc/store/getreviews?l=japanese

r = requests.get(url, params=params)

cursor = r.json()["cursor"]
print(cursor)

json_filename = path.join(path.dirname(
    __file__), "reviewData/" + str(game_id) + ".json")
with open(json_filename, mode='w') as f:
    json.dump(r.json(), f, sort_keys=True, indent=4)
