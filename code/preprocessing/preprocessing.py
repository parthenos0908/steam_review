import json
from os import path
from posixpath import commonpath
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
import re

lemmatizer = WordNetLemmatizer()

isStopwords = True
# "Bug Report, Feature Request, or Just a Rating? On Automatically Classifying App Reviews" RE2015 から引用
CUSTOM_STOPWORDS = ['i', 'me','up','my', 'myself', 'we', 'our', 'ours',
                    'ourselves', 'you', 'your', 'yours','yourself', 'yourselves',
                    'he', 'him', 'his', 'himself', 'she', 'her', 'hers' ,'herself',
                    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
                    'themselves' ,'am', 'is', 'are','a', 'an', 'the', 'and','in',
                    'out', 'on','up','down', 's', 't']

# アルファベット,数字,!,? 以外の文字のみで構成されたトークンを無視
onlyCharacter = True

INPUT_FILENAME = "255710_forum.json"
OUTPUT_FILENAME = "255710_forum.json_cleaned.json"
# F:forum R:review
MODE = "F"

def main():
    input_filepath = path.join(path.dirname(__file__), INPUT_FILENAME)
    input_data = load_json(input_filepath)
    print(len(input_data))
    if MODE == "F":
        preprocessing_data = forumPreprocessing(input_data)
    elif MODE == "R":
        preprocessing_data = reviewPreprocessing(input_data)

    output_filepath = path.join(path.dirname(__file__), OUTPUT_FILENAME)
    save_json(preprocessing_data, output_filepath)


def forumPreprocessing(forums):
    tmp_list = []
    i = 0
    for forum in forums:
        if i % 1000 == 0: print("{0}/{1}".format(i, len(forums)))
        i += 1
        try:
            forum["comment_lem"], commont_wordsNum = lemmatize(forum["comment"])
            forum["title_lem"], title_wordsNum = lemmatize(forum["title"])
            forum["combined"] = forum["title_lem"] + " " + forum["comment_lem"]
            forum["num_words"] = commont_wordsNum + title_wordsNum
            tmp_list.append(forum)
        except Exception as e:
            print(e)
            continue
    return tmp_list

def reviewPreprocessing(reviews):
    tmp_list = []
    i = 0
    for review in reviews:
        if i % 1000 == 0: print("{0}/{1}".format(i, len(reviews)))
        i += 1
        try:
            review["review_lem"], review["num_words"] = lemmatize(review["review"])
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
    main()