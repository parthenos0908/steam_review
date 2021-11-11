import numpy as np
import tensorflow as tf
import transformers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
from os import path
import sys
import collections as cl
import random
from tensorflow.python.keras.utils.vis_utils import plot_model
import numpy
from tqdm import tqdm

# model_nameはここから取得(cf. https://huggingface.co/transformers/pretrained_models.html)
# model_name = "cl-tohoku/bert-base-japanese"
model_name = "bert-base-uncased"
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

MODE = "forum"
ID1 = 255710
ID2 = 227300

INPUT_REVIEW_FILENAME = "data/" + str(ID1) + "/" + str(ID1) + "_review_cleaned_out.json"
INPUT_FORUM_FILENAME = "data/" + str(ID2) + "/" + str(ID2) + "_forum_cleaned.json"

OUTPUT_FILENAME = "data/cross/" + str(ID1) + "_" + str(ID2) + "_" + "predict.json"

# DLパラメータ
num_classes = 3
max_length = 128  #256はメモリ不足
batch_size = 32
epochs = 5
hold_out_rate = 0.7 # 訓練データとテストデータの比率


def main():
    # review読み込み
    input_review_json_filename = path.join(path.dirname(__file__), INPUT_REVIEW_FILENAME)
    r_text_list, r_label_list, r_origin_list = load_review_json(input_review_json_filename)

    # テキストとラベルの関係を維持したままリストをシャッフル
    p = list(zip(r_text_list, r_label_list, r_origin_list))
    random.seed(1)
    random.shuffle(p)
    r_text_list, r_label_list, r_origin_list = zip(*p)

    # 訓練データ(review)
    train_texts = r_text_list[:int(len(r_text_list)*hold_out_rate)]
    train_labels = r_label_list[:int(len(r_label_list)*hold_out_rate)]
    train_texts_origin = r_origin_list[:int(len(r_origin_list)*hold_out_rate)] # 現状不要

    # テストデータ
    test_texts = r_text_list[int(len(r_text_list)*hold_out_rate):]
    test_labels = r_label_list[int(len(r_label_list)*hold_out_rate):]
    test_texts_origin = r_origin_list[int(len(r_origin_list)*hold_out_rate):]

    # 教師データにforumを使う場合
    if MODE == "forum":

        input_forum_json_filename = path.join(path.dirname(__file__), INPUT_FORUM_FILENAME)
        f_text_list, f_label_list = load_forum_json(input_forum_json_filename)

        # forumのデータ数をreviewのtestデータ数に合わせて学習--------------------------------------
        # forum = load_forum_json(input_forum_json_filename)
        # br = []
        # fr = []
        # other = []
        # for f in forum:
        #     if forum[1] == 0:
        #         br.append(forum[0])
        #     elif forum[1] == 1:
        #         fr.append(forum[0])
        #     elif forum[1] == 2:
        #         other.append(forum[0])
        # f_text_list = random.sample(br, k=train_texts.count(0)) + random.sample(fr, k=train_texts.count(1)) + random.sample(other, k=train_texts.count(2))
        # f_label_list = []
        # for _ in range(train_texts.count(0)):
        #     f_label_list.append(0)
        # for _ in range(train_texts.count(1)):
        #     f_label_list.append(1)
        # for _ in range(train_texts.count(2)):
        #     f_label_list.append(2)
        #---------------------------------------------------------------------------------------------

        p = list(zip(f_text_list, f_label_list))
        random.seed(1)
        random.shuffle(p)
        f_text_list, f_label_list = zip(*p)
        train_texts = f_text_list
        train_labels = f_label_list

    print("[train data] BR:{0}, FR:{1}, OTHER:{2} [test data] BR:{3}, FR:{4}, OTHER:{5}".format(
        train_labels.count(0), train_labels.count(1), train_labels.count(2), test_labels.count(0), test_labels.count(1), test_labels.count(2)))


    # モデル構築
    # x_:テキストデータ, y_:ラベル
    x_train = to_features(train_texts, max_length)
    y_train = tf.keras.utils.to_categorical(
        train_labels, num_classes=num_classes)
    model = build_model(model_name, num_classes=num_classes,
                        max_length=max_length)

    # 訓練
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    # 予測
    x_test = to_features(test_texts, max_length)
    y_test = np.asarray(test_labels)
    y_preda = model.predict(x_test)
    y_pred = np.argmax(y_preda, axis=1)
    print("Accuracy: %.5f" % accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))

    output_json_filename = path.join(path.dirname(__file__), OUTPUT_FILENAME)
    output_json(output_json_filename, test_texts, y_test, y_pred, test_texts_origin)


# テキストのリストをtransformers用の入力データに変換

def to_features(texts, max_length):
    shape = (len(texts), max_length)
    # input_idsやattention_mask, token_type_idsの説明はglossaryに記載(cf. https://huggingface.co/transformers/glossary.html)
    input_ids = np.zeros(shape, dtype="int32")
    attention_mask = np.zeros(shape, dtype="int32")
    token_type_ids = np.zeros(shape, dtype="int32")
    for i, text in enumerate(tqdm(texts)):
        encoded_dict = tokenizer.encode_plus(
            text, max_length=max_length, pad_to_max_length=True)
        input_ids[i] = encoded_dict["input_ids"]
        attention_mask[i] = encoded_dict["attention_mask"]
        token_type_ids[i] = encoded_dict["token_type_ids"]
    return [input_ids, attention_mask, token_type_ids]

# 単一テキストをクラス分類するモデルの構築


def build_model(model_name, num_classes, max_length):
    input_shape = (max_length, )
    input_ids = tf.keras.layers.Input(input_shape, dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(input_shape, dtype=tf.int32)
    token_type_ids = tf.keras.layers.Input(input_shape, dtype=tf.int32)
    bert_model = transformers.TFBertModel.from_pretrained(model_name)
    last_hidden_state, pooler_output = bert_model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )
    # transformersのバージョンによってはエラーが起きる．ver2.11.0で実行
    output = tf.keras.layers.Dense(
        num_classes, activation="softmax")(pooler_output)
    model = tf.keras.Model(
        inputs=[input_ids, attention_mask, token_type_ids], outputs=[output])
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy", metrics=["acc"])
    return model

# jsonを読み込んでリストに格納


def load_review_json(json_filename):
    text_list = []
    label_list = []
    origin_list = []
    with open(json_filename, mode='r') as f:
        json_data = json.load(f)
    for text in json_data:
        if ("label" in text) and (text["label"] in [0,1,2]):
            text_list.append(text['review_lem'])
            label_list.append(text['label'])
            origin_list.append(text['review'])
    return text_list, label_list, origin_list


def load_forum_json(json_filename):
    text_list = []
    label_list = []
    with open(json_filename, mode='r') as f:
        json_data = json.load(f)
    for text in json_data:
        if text["num_words"] <= max_length:
            text_list.append(text['combined'])
            label_list.append(text['label'])
    return text_list, label_list

# 予測結果をjsonに書き込み


def output_json(json_filename, comment_list, answer_list, pred_list, origin_list):
    output = []
    for i in range(len(comment_list)):
        data = cl.OrderedDict()
        data["review_lem"] = str(comment_list[i])
        data["review"] = str(origin_list[i])
        data["answer"] = int(answer_list[i])
        data["pred"] = int(pred_list[i])
        output.append(data)

    with open(json_filename, mode='w') as f:
        json.dump(output, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    main()