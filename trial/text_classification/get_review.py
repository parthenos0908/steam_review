import requests
import json
from os import path

game_id = 945360
url = "https://store.steampowered.com/appreviews/" + str(game_id) + "?json=1"
params = {
    "start_offset":0
    }

r = requests.get(url, params=params)

json_filename = path.join(path.dirname(__file__), "reviewData/" + str(game_id) + ".json")
with open(json_filename, mode='w') as f:
    json.dump(r.json(), f, sort_keys=True, indent=4)