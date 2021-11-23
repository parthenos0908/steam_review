import json
from os import path

ID = 255710  # Cities
ID = 227300  # Eur

INPUT_FILENAME = "{0}/{0}_forum_cleaned.json".format(ID)
INPUT_FILENAME = "{0}/{0}_review.json".format(ID)


def main():
    input_filepath = path.join(path.dirname(__file__), INPUT_FILENAME)
    forum = load_json(input_filepath)

    count_bug = 0
    count_feature = 0
    count_general = 0
    for i, data in enumerate(forum):
        if data.get('label') in (0, 1, 2):
            if data["label"] == 0:
                count_bug += 1
            elif data["label"] == 1:
                count_feature += 1
            elif data["label"] == 2:
                count_general += 1
    sum = count_bug + count_feature + count_general
    print("bug:{0}, feature:{1}, other:{2}, sum:{3}".format(
        count_bug, count_feature, count_general, sum))


def load_json(json_filepath):
    with open(json_filepath, mode='r') as f:
        json_data = json.load(f)
    return json_data


def save_json(output_list, json_filepath):
    with open(json_filepath, mode='w') as f:
        json.dump(output_list, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    main()
