import requests
import json

url = "https://store.steampowered.com/appreviews/945360?json=1"
params = {"start_offset":0}

r = requests.get(url, params=params)

print(r.json())