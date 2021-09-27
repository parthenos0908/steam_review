import numpy as np
import tensorflow as tf
import transformers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
from os import path
import collections as cl
import random
from tensorflow.python.keras.utils.vis_utils import plot_model
import numpy
from tqdm import tqdm


# model_nameはここから取得(cf. https://huggingface.co/transformers/pretrained_models.html)
# model_name = "cl-tohoku/bert-base-japanese"
model_name = "bert-base-uncased"
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

INPUT_REVIEW_FILENAME = "255710_review_cleaned_out.json"
INPUT_FORUM_FILENAME = "255710_forum_cleaned.json"

OUTPUT_FILENAME = "255710_predict_forum.json"



KEY = ""

MODE = ""

# DLパラメータ
num_classes = 3
max_length = 128  #256はメモリ不足
batch_size = 32
epochs = 5


def main():
    # review読み込み
    input_review_json_filename = path.join(path.dirname(__file__), INPUT_REVIEW_FILENAME)
    r_text_list, r_label_list, r_origin_list = load_review_json(input_review_json_filename)

    # テキストとラベルの関係を維持したままリストをシャッフル
    p = list(zip(r_text_list, r_label_list, r_origin_list))
    random.seed(1)
    random.shuffle(p)
    r_text_list, r_label_list, r_origin_list = zip(*p)

    # 訓練データ
    train_texts = r_text_list[:int(len(r_text_list)*0.7)]
    train_labels = r_label_list[:int(len(r_label_list)*0.7)]
    train_texts_origin = r_origin_list[:int(len(r_origin_list)*0.7)] # 現状不要

    # テストデータ
    test_texts = r_text_list[int(len(r_text_list)*0.7):]
    test_labels = r_label_list[int(len(r_label_list)*0.7):]
    test_texts_origin = r_origin_list[int(len(r_origin_list)*0.7):]

    # 教師データにforumを使う場合
    if MODE == "F":
        input_forum_json_filename = path.join(path.dirname(__file__), INPUT_FORUM_FILENAME)
        f_text_list, f_label_list = load_forum_json(input_forum_json_filename)
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
        # if i == 724: continue
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
        if "label" in text:
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
    # text = "weird behavior of tourist who arrive via train tl dr tourist often arrive at wrong station when come via rail have 3 train station city which will call b c for convenience all three connect to train line intercity train allow for all of outermost station all inbound train either go through or pass next to before enter rest of city do n't know when begin to happen for while thing all look normal each of station have 100-300 passenger weekly but a begin build overpass for cim other side of station to reach station notice that even though number indicate moderate usage platform fill with busy cim no one actually enter or exit station 's door upon close investigation find that everyone platform literally everyone tourist or new immigrant to city who wait for city train to b or c even though intercity train can do arrive at b c. there explanation to phenomenon ? do intercity train arrive at random passenger pick destination randomly after alight ?"

    # shape = (1, max_length)
    # # input_idsやattention_mask, token_type_idsの説明はglossaryに記載(cf. https://huggingface.co/transformers/glossary.html)
    # input_ids = np.zeros(shape, dtype="int32")
    # attention_mask = np.zeros(shape, dtype="int32")
    # token_type_ids = np.zeros(shape, dtype="int32")
    # encoded_dict = tokenizer.encode_plus(
    #     text, max_length=max_length, pad_to_max_length=True)
    # print(encoded_dict)
    main()
