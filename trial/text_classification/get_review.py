import requests
import json
import urllib.parse
import time
from os import path


def main():
    appid = 1172470

    reviews = get_all_reviews(appid)
    print(len(reviews))

    json_filename = path.join(path.dirname(
        __file__), "reviewData/" + str(appid) + ".json")
    with open(json_filename, mode='w') as f:
        json.dump(reviews, f, sort_keys=True, indent=4)


def get_reviews(appid, params={'json': 1}):
    url = 'https://store.steampowered.com/appreviews/'
    response = requests.get(url=url+str(appid), params=params,
                            headers={'User-Agent': 'Mozilla/5.0'})
    print(response.url)
    return response.json()


def get_all_reviews(appid):
    reviews = []
    cursor = '*'
    params = {
        'json': 1,
        'filter': 'all',
        'language': 'english',
        # 'day_range': 9223372036854775807,  # n日前までのreviewを取得(bigintの最大値を入力)
        'day_range': 1,
        'review_type': 'all',
        'purchase_type': 'all',
        "num_per_page": 100,
        # "start_offset": 0,
    }

    while (1):
        params['cursor'] = cursor.encode()

        response = get_reviews(appid, params)
        cursor = response['cursor']
        reviews += response['reviews']
        print("params:{0} coursor:{1}".format(params["cursor"], cursor))
        print("--------------------")
        time.sleep(0.1)

        if len(response['reviews']) < params["num_per_page"]:
            break

    return reviews


if __name__ == '__main__':
    main()
