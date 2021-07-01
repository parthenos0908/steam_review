import enum
import json
from os import path

INPUT_FILENAME = "f_in.json"
OUTPUT_FILENAME = "f_out.json"

BR_WORDS = ["bug", "crash", "fix"]
FR_WORDS = [""]


def main():
    input_filepath = path.join(path.dirname(__file__), INPUT_FILENAME)
    data = load_json(input_filepath)

    BR_review = []
    for i, datum in enumerate(data):
        flag = False
        for br_word in BR_WORDS:
            if br_word in datum["review_lem"]:
                flag = True
        if flag:
            BR_review.append(datum)

    print(len(BR_review))
    output_filepath = path.join(path.dirname(__file__), OUTPUT_FILENAME)
    save_json(BR_review, output_filepath)


def load_json(json_filepath):
    with open(json_filepath, mode='r') as f:
        json_data = json.load(f)
    return json_data


def save_json(output_list, json_filepath):
    with open(json_filepath, mode='w') as f:
        json.dump(output_list, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    main()
