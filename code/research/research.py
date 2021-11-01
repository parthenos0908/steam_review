import json
from os import path
import collections

from scipy.sparse import data
import seaborn as sns
import matplotlib.pyplot as plt

ID = [255710 ,227300, 427520]

FORUM_FILENAMES = list(map(lambda x:"../../data/" + str(x) + "/" + str(x) + "_forum_cleaned.json", ID))
REVIEW_FILENAMES = list(map(lambda x:"../../data/" + str(x) + "/" + str(x) + "_review_cleaned_out.json", ID))

LABELS = ["review\n(Cities: Skylines)", "review\n(Euro Truck Sim)", "review\n(Factorio)", "forum\n(Cities: Skylines)", "forum\n(Euro Truck Sim)", "forum\n(Factorio)"]

max_length = 128

def main():
    forums = []
    for f in FORUM_FILENAMES:
        forums.append(load_json(path.join(path.dirname(__file__), f)))
    reviews = []
    for r in REVIEW_FILENAMES:
        reviews.append(load_json(path.join(path.dirname(__file__), r)))

    # 単語数の分布
    forums_num_words = []
    for i, forum in enumerate(forums):
        tmp = []
        sum = 0
        for topics in forum:
            tmp.append(topics["num_words"])
            sum += topics["num_words"]
        forums_num_words.append(tmp)
        forum_average = sum / len(forum)
        print("[forum{0}]:{1}".format(ID[i], forum_average))

    reviews_num_words = []
    for i, review in enumerate(reviews):
        tmp = []
        sum = 0
        for topics in review:
            tmp.append(topics["num_words"])
            sum += topics["num_words"]
        reviews_num_words.append(tmp)
        review_average = sum / len(review)
        print("[review{0}]:{1}".format(ID[i], review_average))

    plt.figure(figsize=(10,6))
    df = reviews_num_words + forums_num_words
    plt.boxplot(df, labels=LABELS, sym="", showmeans=True)
    plt.show()

    # 頻出語彙
    # plt.rcParams["font.size"] = 18
    # for i, forum in enumerate(forums):
    #     forum_word_list = []
    #     for topics in forum:
    #         if topics["num_words"] <= max_length:
    #             forum_word_list.extend(topics["combined"].split())
    #     c = collections.Counter(forum_word_list).most_common(20)
    #     # print(c.most_common(20))
    #     df = [val[0] for val in c]
    #     label = [val[1] for val in c]
    #     plt.title(LABELS[i], fontsize=16)
    #     plt.barh(df[::-1], label[::-1])
    #     plt.show()


    # for i, review in enumerate(reviews):
    #     review_word_list = []
    #     for topics in review:
    #         if topics["num_words"] <= max_length:
    #             review_word_list.extend(topics["review_lem"].split())
    #     c = collections.Counter(review_word_list).most_common(20)
    #     # print(c.most_common(20))
    #     df = [val[0] for val in c]
    #     label = [val[1] for val in c]
    #     plt.title(LABELS[i+3], fontsize=16)
    #     plt.barh(df[::-1], label[::-1])
    #     plt.show()


def load_json(json_filepath):
    with open(json_filepath, mode='r') as f:
        json_data = json.load(f)
    return json_data


def save_json(output_list, json_filepath):
    with open(json_filepath, mode='w') as f:
        json.dump(output_list, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    main()
