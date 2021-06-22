import json
from os import path
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()


def main():
    input_json_filename = path.join(path.dirname(__file__), "test.json")
    input_data = load_json(input_json_filename)
    preprocessing_data_list = []
    for forum in input_data:
        word_tokenize_list = word_tokenize(forum["comment"])
        print(word_tokenize_list)

        token_list = [lemmatizer.lemmatize(w).lower()
                      for w in word_tokenize_list]
        forum["Lemmatization"] = " ".join(token_list)

        preprocessing_data_list.append(forum)
    output_json_filename = path.join(
        path.dirname(__file__), "preprocessing_test.json")
    save_json(preprocessing_data_list, output_json_filename)


def load_json(json_filename):
    with open(json_filename, mode='r') as f:
        json_data = json.load(f)
    return json_data


def save_json(output_list, json_filename):
    with open(json_filename, mode='w') as f:
        json.dump(output_list, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    main()
