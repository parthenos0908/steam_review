import enum
import json
import random
from os import path

INPUT_FILENAME = "review/227300_review_cleaned_out.json"
OUTPUT = "review/227300_review"

# BR_WORDS = ["bug", "fix", "problem", "issue", "defect", "crash", "solve"]
BR_WORDS = ["bug", "fix", "crash"]
# FR_WORDS = ["add", "please", "could", "would", "hope", "improve", "miss",
#             "need", "prefer", "request", "should", "suggest", "want", "wish"]
FR_WORDS = ["please", "hope", "improve", "need",
            "prefer", "request", "suggest", "wish"]

# random:ランダムに5000件, BR:バグ報告, FR:機能要求
MODE = "FR"


def main():
    input_filepath = path.join(path.dirname(__file__), INPUT_FILENAME)
    data = load_json(input_filepath)

    random.seed(0)
    if MODE == "random":
        filtered_review = random.sample(data, 5000)
    elif MODE == "BR":
        filtered_review = filter_BR_review(data)
        random.shuffle(filtered_review)
    elif MODE == "FR":
        filtered_review = filter_FR_review(data)
        random.shuffle(filtered_review)
    output_filename = OUTPUT + "_" + MODE + \
        "_" + str(len(filtered_review)) + ".json"
    print(output_filename)

    output_filepath = path.join(path.dirname(__file__), output_filename)
    save_json(filtered_review, output_filepath)


def filter_BR_review(data):
    filtered_review = []
    for i, datum in enumerate(data):
        flag = False
        for br_word in BR_WORDS:
            if br_word in datum["review_lem"]:
                flag = True
        if flag:
            filtered_review.append(datum)
    return filtered_review


def filter_FR_review(data):
    filtered_review = []
    for i, datum in enumerate(data):
        flag = False
        for fr_word in FR_WORDS:
            if fr_word in datum["review_lem"]:
                flag = True
        if flag:
            filtered_review.append(datum)
    return filtered_review


def load_json(json_filepath):
    with open(json_filepath, mode='r') as f:
        json_data = json.load(f)
    return json_data


def save_json(output_list, json_filepath):
    with open(json_filepath, mode='w') as f:
        json.dump(output_list, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    main()
