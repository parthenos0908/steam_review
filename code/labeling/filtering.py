import enum
import json
from os import path

INPUT_FILENAME = "review/255710_review_cleaned.json"
OUTPUT_FILENAME = "f_out.json"

# BR_WORDS = ["bug", "fix", "problem", "issue", "defect", "crash", "solve"]
BR_WORDS = ["bug", "fix", "crash"]
# FR_WORDS = ["add", "please", "could", "would", "hope", "improve", "miss",
#             "need", "prefer", "request", "should", "suggest", "want", "wish"]
FR_WORDS = ["please", "hope", "improve", "need",
            "prefer", "request", "suggest", "wish"]


def main():
    input_filepath = path.join(path.dirname(__file__), INPUT_FILENAME)
    data = load_json(input_filepath)

    count_filtered_review(data)
    filtered_review = filter_review(data)
    print(len(filtered_review))

    output_filepath = path.join(path.dirname(__file__), OUTPUT_FILENAME)
    save_json(filtered_review, output_filepath)


def filter_review(data):
    filtered_review = []
    for i, datum in enumerate(data):
        flag = False
        for fr_word in FR_WORDS:
            if fr_word in datum["review_lem"]:
                flag = True
        if flag:
            filtered_review.append(datum)
    return filtered_review


def count_filtered_review(data):
    print("all data:{0}".format(len(data)))
    for fr_word in FR_WORDS:
        filtered_review = []
        for i, datum in enumerate(data):
            if fr_word in datum["review_lem"]:
                filtered_review.append(datum)
        print("{0}:{1}".format(fr_word, len(filtered_review)))


def load_json(json_filepath):
    with open(json_filepath, mode='r') as f:
        json_data = json.load(f)
    return json_data


def save_json(output_list, json_filepath):
    with open(json_filepath, mode='w') as f:
        json.dump(output_list, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    main()
