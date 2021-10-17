from itertools import count
import json
from os import path
from posixpath import commonpath
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
import re
from tqdm import tqdm

from nltk.util import pr

lemmatizer = WordNetLemmatizer()

isStopwords = True
# "Bug Report, Feature Request, or Just a Rating? On Automatically Classifying App Reviews" RE2015 から引用
CUSTOM_STOPWORDS = ['i', 'me', 'up', 'my', 'myself', 'we', 'our', 'ours',
                    'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
                    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                    'it', 'its', 'itself', 'this', 'they', 'them', 'their', 'theirs',
                    'themselves', 'am', 'is', 'are', 'a', 'an', 'the', 'and', 'in',
                    'out', 'on', 'up', 'down', 's', 't']

# アルファベット,数字,!,? 以外の文字のみで構成されたトークンを無視
onlyCharacter = True

LABEL = {
    "Bug": 0,
    "Feature": 1,
    "General": 2
}

INPUT_FILENAME = "227300_review.json"
OUTPUT_FILENAME = "227300_review_cleaned.json"
# F:forum R:review
MODE = "R"


def main():
    input_filepath = path.join(path.dirname(__file__), INPUT_FILENAME)
    input_data = load_json(input_filepath)
    print(len(input_data))
    if MODE == "F":
        preprocessing_data = forumPreprocessing(input_data)
        count_br = 0
        count_fr = 0
        count_other = 0
        for p in preprocessing_data:
            if p["label"] == 0:
                count_br += 1
            elif p["label"] == 1:
                count_fr += 1
            elif p["label"] == 2:
                count_other += 1
        print("bug:{0}, feature:{1}, other:{2}".format(
            count_br, count_fr, count_other))
    elif MODE == "R":
        preprocessing_data = reviewPreprocessing(input_data)

    output_filepath = path.join(path.dirname(__file__), OUTPUT_FILENAME)
    save_json(preprocessing_data, output_filepath)


def forumPreprocessing(forums):
    tmp_list = []
    i = 0
    for i, forum in enumerate(tqdm(forums)):
        try:
            forum["comment_lem"], commont_wordsNum = lemmatize(
                forum["comment"])
            forum["title_lem"], title_wordsNum = lemmatize(forum["title"])
            forum["combined"] = forum["title_lem"] + " " + forum["comment_lem"]
            forum["num_words"] = commont_wordsNum + title_wordsNum
            forum["label"] = LABEL[forum["label"]]
            forum["id"] = i
            tmp_list.append(forum)
        except Exception as e:
            print(e)
            continue
    return tmp_list


def reviewPreprocessing(reviews):
    tmp_list = []
    for i, review in enumerate(tqdm(reviews)):
        try:
            review["review_lem"], review["num_words"] = lemmatize(
                review["review"])
            review["id"] = i
            review["label"] = -1
            tmp_list.append(review)
        except Exception as e:
            print(e)
            continue
    return tmp_list

# (sentence) → (sentence, 単語数)


def lemmatize(text):
    word_tokenize_list = pos_tag(word_tokenize(text))
    token_list = []
    for word, tag in word_tokenize_list:
        if ((word.lower() in CUSTOM_STOPWORDS) and isStopwords) or (re.fullmatch(r'[^a-zA-Z0-9!?]+', word) and onlyCharacter):
            continue
        wn_tag = pos_tagger(tag)
        if wn_tag is None:
            token_list.append(lemmatizer.lemmatize(word.lower()))
        else:
            token_list.append(lemmatizer.lemmatize(word.lower(), wn_tag))
    return " ".join(token_list), len(token_list)

# tagの変換 (Stanford POS → WordNet POS)


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wn.ADJ
    elif nltk_tag.startswith('V'):
        return wn.VERB
    elif nltk_tag.startswith('N'):
        return wn.NOUN
    elif nltk_tag.startswith('R'):
        return wn.ADV
    else:
        return None


def load_json(json_filepath):
    with open(json_filepath, mode='r') as f:
        json_data = json.load(f)
    return json_data


def save_json(output_list, json_filepath):
    with open(json_filepath, mode='w') as f:
        json.dump(output_list, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    # text = "I bought this game yesterday and it is full of bugs!"
    # print(word_tokenize(text))
    # print(lemmatize(text))
    main()
