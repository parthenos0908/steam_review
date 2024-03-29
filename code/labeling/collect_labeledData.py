import json
from os import path
from tqdm import tqdm  # 進捗表示

LABELED_REVIEW = "review/227300_review_FR_5813_out.json"
ALL_REVIEW = "review/227300_review_cleaned_out.json"

# random:ランダムに5000件, BR:バグ報告, FR:機能要求
MODE = "FR"

# BR,FRのotherは排除
if MODE == "random":
    LABELLIST = [0, 1, 2, 3]
elif MODE == "BR" or MODE == "FR":
    LABELLIST = [0, 1, 3]


def main():
    data = []
    input_filepath = path.join(path.dirname(__file__), LABELED_REVIEW)
    labeled_reviews = load_json(input_filepath)
    for review in labeled_reviews:
        if review["label"] in LABELLIST:
            data.append(review)

    input_filepath = path.join(path.dirname(__file__), ALL_REVIEW)
    all_reviews = load_json(input_filepath)
    for datum in tqdm(data):
        all_reviews[datum["id"]]["label"] = datum["label"]

    output_filepath = path.join(path.dirname(__file__), ALL_REVIEW)
    save_json(all_reviews, output_filepath)


def load_json(json_filepath):
    with open(json_filepath, mode='r') as f:
        json_data = json.load(f)
    return json_data


def save_json(output_list, json_filepath):
    with open(json_filepath, mode='w') as f:
        json.dump(output_list, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    main()
