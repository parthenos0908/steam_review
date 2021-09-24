import json
from os import path

INPUT_FILENAMES = [
    "review/255710_review_cleaned_5000_out.json",
    "review/255710_review_BR_filtered_3757_out.json",
    "review/255710_review_FR_filtered_7464_out.json",    
]

OUTPUT_FILENAME = "review/255710_review_cleaned_out.json"

def main():
    # inputfileのjsonを辞書listにして結合
    data = []
    # random
    input_filepath = path.join(path.dirname(__file__), INPUT_FILENAMES[0])
    random = load_json(input_filepath)
    for review in random:
        if "label" in review:
            if review["label"] in [0,1,2]:
                data.append(review)
    # BR
    input_filepath = path.join(path.dirname(__file__), INPUT_FILENAMES[1])
    br = load_json(input_filepath)
    for review in br:
        if "label" in review:
            if review["label"] in [0,1]:
                data.append(review)
    # FR
    input_filepath = path.join(path.dirname(__file__), INPUT_FILENAMES[2])
    fr = load_json(input_filepath)
    for review in fr:
        if "label" in review:
            if review["label"] in [0,1]:
                data.append(review)
    print("labeled_review:{0}".format(len(data)))

    recommendationid_list = []
    for datum in data:
        recommendationid_list.append(datum["recommendationid"])
    print("独立review数:{0}".format(len(set(recommendationid_list))))

    dup_id_list = []
    for id in set(recommendationid_list):
        dup = recommendationid_list.count(id)
        if dup > 1:
            dup_id_list.append(id)
    print("重複review数:{0}".format(len(dup_id_list)))

    # 同一reviewに異なるラベルを付けていないかチェック    
    for dup_id in dup_id_list:
        labelbox = []
        for datum in data:
            if datum["recommendationid"] == dup_id:
                labelbox.append(datum["label"])
        if len(set(labelbox)) > 1:
            print("{0}: {1}".format(dup_id, labelbox))

    input_filepath = path.join(path.dirname(__file__), "review/255710_review_cleaned.json")
    allReview = load_json(input_filepath)

    # 全reviewデータにまとめる
    for datum in data:
        for i, review in enumerate(allReview):
            if datum["recommendationid"] == allReview[i]["recommendationid"]:
                allReview[i]["label"] = datum["label"]

    count_br = 0
    count_fr = 0
    count_other = 0
    for i, review in enumerate(allReview):
        if "label" in review:
            if allReview[i]["label"] == 0:
                count_br += 1
            elif allReview[i]["label"] == 1:
                count_fr += 1
            elif allReview[i]["label"] == 2:
                count_other += 1
    print("bug:{0}, feature:{1}, other:{2}".format(count_br, count_fr, count_other))

    output_filepath = path.join(path.dirname(__file__), OUTPUT_FILENAME)
    save_json(allReview, output_filepath)

def load_json(json_filepath):
    with open(json_filepath, mode='r') as f:
        json_data = json.load(f)
    return json_data


def save_json(output_list, json_filepath):
    with open(json_filepath, mode='w') as f:
        json.dump(output_list, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    main()
