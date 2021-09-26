import numpy as np
import tensorflow as tf
import transformers
from sklearn.metrics import accuracy_score, classification_report
import json
from os import path
import collections as cl
import random
from tensorflow.python.keras.utils.vis_utils import plot_model
import numpy


# model_nameはここから取得(cf. https://huggingface.co/transformers/pretrained_models.html)
# model_name = "cl-tohoku/bert-base-japanese"
model_name = "bert-base-uncased"
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

INPUT_FILENAME = "all.json"
OUTPUT_FILENAME = "output.json"
KEY = ""

# DLパラメータ
num_classes = 4
max_length = 64
batch_size = 32  # 24でメモリ不足
epochs = 10


def main():
    input_json_filename = path.join(path.dirname(__file__), INPUT_FILENAME)
    text_list, label_list = load_json(input_json_filename)

    # labelは文字列なので数値に変換
    label_number_dict = {'Bug': 0, 'Rating': 1,
                         'Feature': 2, 'UserExperience': 3}
    label_number_list = []
    for label in label_list:
        if label in label_number_dict:
            label_number_list.append(label_number_dict[label])
        else:
            label_number_list.append(-1)

    # テキストとラベルの関係を維持したままリストをシャッフル
    p = list(zip(text_list, label_number_list))
    random.shuffle(p)
    text_list, label_number_list = zip(*p)

    # 訓練データ
    train_texts = text_list[:int(len(text_list)*0.7)]
    train_labels = label_number_list[:int(len(label_number_list)*0.7)]

    # テストデータ
    test_texts = text_list[int(len(text_list)*0.7):]
    test_labels = label_number_list[int(len(label_number_list)*0.7):]

    print("train data:{0}, test data:{1}".format(
        len(train_texts), len(test_texts)))

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

    output_json_filename = path.join(path.dirname(__file__), OUTPUT_FILENAME)
    output_json(output_json_filename, test_texts, y_test, y_pred)


# テキストのリストをtransformers用の入力データに変換

def to_features(texts, max_length):
    shape = (len(texts), max_length)
    # input_idsやattention_mask, token_type_idsの説明はglossaryに記載(cf. https://huggingface.co/transformers/glossary.html)
    input_ids = np.zeros(shape, dtype="int32")
    attention_mask = np.zeros(shape, dtype="int32")
    token_type_ids = np.zeros(shape, dtype="int32")
    for i, text in enumerate(texts):
        encoded_dict = tokenizer.encode_plus(
            text, max_length=max_length, pad_to_max_length=True)
        input_ids[i] = encoded_dict["input_ids"]
        attention_mask[i] = encoded_dict["attention_mask"]
        token_type_ids[i] = encoded_dict["token_type_ids"]
    print([input_ids, attention_mask, token_type_ids])
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


def load_json(json_filename):
    text_list = []
    label_list = []
    with open(json_filename, mode='r') as f:
        json_data = json.load(f)
    for text in json_data:
        text_list.append(text['stopwords_removal_lemmatization'])
        label_list.append(text['label'])
    return text_list, label_list

# 予測結果をjsonに書き込み


def output_json(json_filename, comment_list, answer_list, pred_list):
    output = []
    for i in range(len(comment_list)):
        data = cl.OrderedDict()
        data["stopwords_removal_lemmatization"] = str(comment_list[i])
        data["answer"] = int(answer_list[i])
        data["pred"] = int(pred_list[i])
        output.append(data)

    with open(json_filename, mode='w') as f:
        json.dump(output, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    model = build_model(model_name, num_classes=num_classes,
                        max_length=max_length)
    plot_model(model)
    # main()
