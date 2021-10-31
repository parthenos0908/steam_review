import json
from os import path
import matplotlib.pyplot as plt


FORUM_FILENAME = "427520_forum_cleaned.json"
REVIEW_FILENAME = "427520_review_cleaned_out.json"


def main():
    forum_filepath = path.join(path.dirname(__file__), FORUM_FILENAME)
    forums = load_json(forum_filepath)
    review_filepath = path.join(path.dirname(__file__), REVIEW_FILENAME)
    reviews = load_json(review_filepath)
    forum_num_words = []
    for forum in forums:
        forum_num_words.append(forum["num_words"])
    review_num_words = []
    for review in reviews:
        review_num_words.append(review["num_words"])

    sum = 0
    for num_words in forum_num_words:
        sum += num_words
    forum_average = sum / len(forum_num_words)
    print(forum_average)

    sum = 0
    for num_words in review_num_words:
        sum += num_words
    review_average = sum / len(review_num_words)
    print(review_average)


def load_json(json_filepath):
    with open(json_filepath, mode='r') as f:
        json_data = json.load(f)
    return json_data


def save_json(output_list, json_filepath):
    with open(json_filepath, mode='w') as f:
        json.dump(output_list, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    main()
