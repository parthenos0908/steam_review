import numpy as np
import tensorflow as tf
import transformers
import json
from os import path
import sys
import collections as cl
import random
from tensorflow.python.keras.utils.vis_utils import plot_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
import numpy
from tqdm import tqdm
import matplotlib.pyplot as plt

# model_nameはここから取得(cf. https://huggingface.co/transformers/pretrained_models.html)
# model_name = "cl-tohoku/bert-base-japanese"
model_name = "bert-base-uncased"
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

# DLパラメータ
num_classes = 3
max_length = 128  # 256はメモリ不足
batch_size = 32
epochs = 5
hold_out_rate = 0.7  # 訓練データとテストデータの比率


def main():
    # review読み込み
    input_review_json_filename = path.join(
        path.dirname(__file__), INPUT_REVIEW_FILENAME)
    r_text_list, r_label_list, r_origin_list = load_review_json(
        input_review_json_filename)

    # テキストとラベルの関係を維持したままリストをシャッフル
    p = list(zip(r_text_list, r_label_list, r_origin_list))
    random.seed(1)
    random.shuffle(p)
    r_text_list, r_label_list, r_origin_list = zip(*p)

    # 訓練データ(review)
    train_texts = r_text_list[:int(len(r_text_list)*hold_out_rate)]
    train_labels = r_label_list[:int(len(r_label_list)*hold_out_rate)]
    train_texts_origin = r_origin_list[:int(
        len(r_origin_list)*hold_out_rate)]  # 現状不要

    # テストデータ
    test_texts = r_text_list[int(len(r_text_list)*hold_out_rate):]
    test_labels = r_label_list[int(len(r_label_list)*hold_out_rate):]
    test_texts_origin = r_origin_list[int(len(r_origin_list)*hold_out_rate):]

    # 教師データにforumを使う場合
    if MODE == "forum":

        input_forum_json_filename = path.join(
            path.dirname(__file__), INPUT_FORUM_FILENAME)
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
        # ---------------------------------------------------------------------------------------------

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

    output_model_filename = path.join(
        path.dirname(__file__), MODEL_WEIGHT_FILENAME)

    # 新しく学習する(0)or既存の学習結果使う(1)
    tmp = 1
    if tmp == 0:
        # 訓練
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

        # モデルの保存
        model.save_weights(output_model_filename)
    elif tmp == 1:
        # モデルの読み込み
        model.load_weights(output_model_filename)

    # 予測
    x_test = to_features(test_texts, max_length)
    y_test = np.asarray(test_labels)
    y_test_b = label_binarize(y_test, classes=[0, 1, 2])
    y_preda = model.predict(x_test)
    y_pred = np.argmax(y_preda, axis=1)

    
    true_list = [[], [], []]
    for label in y_test_b:
        true_list[0].append(label[0])
        true_list[1].append(label[1])
        true_list[2].append(label[2])

    score_list = [[], [], []]
    for score in y_preda:
        score_list[0].append(score[0])
        score_list[1].append(score[1])
        score_list[2].append(score[2])

    plot_roc(true_list, score_list)
    plot_pr(true_list, score_list)

    print("Accuracy: %.5f" % accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))

    output_json_filename = path.join(path.dirname(__file__), OUTPUT_FILENAME)
    output_json(output_json_filename, test_texts,
                y_test, y_pred, test_texts_origin)

def plot_roc(true_list, score_list, TAGS=["バグ報告", "機能要求", "その他　"], COLOR = ["#d62728", "#2ca02c", "#1f77b4"], fontsize=16):
    plt.rcParams["font.size"] = fontsize
    plt.rcParams['pdf.fonttype'] = 42 #Type3フォント回避
    for i in range(len(TAGS)):
        fpr, tpr, thresholds = roc_curve(true_list[i], score_list[i])
        label = "{0}：AUC = {1:.2f}".format(TAGS[i], roc_auc_score(true_list[i], score_list[i]))
        plt.plot(fpr, tpr, label=label, color=COLOR[i])
        plt.xlabel('FPR: False positive rate')
        plt.ylabel('TPR: True positive rate')
        plt.ylim([0, 1])
        plt.grid()
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.legend(loc="lower right", prop={"family":"meiryo"})
    plt.tight_layout()
    roc_filename = path.join(path.dirname(__file__), ROC_FILENAME + ".pdf")
    plt.savefig(roc_filename)
    plt.clf()  # 描画リセット

def plot_pr(true_list, score_list, TAGS=["バグ報告", "機能要求", "その他　"], COLOR = ["#d62728", "#2ca02c", "#1f77b4"], fontsize=16):
    plt.rcParams["font.size"] = fontsize
    plt.rcParams['pdf.fonttype'] = 42 #Type3フォント回避
    plt.tight_layout()
    for i in range(len(TAGS)-1):
        precision, recall, thresholds = precision_recall_curve(
            true_list[i], score_list[i])
        label = "[{0}] AUC = {1:.2f}".format(TAGS[i], auc(recall, precision))
        plt.plot(recall, precision, label=label, color=COLOR[i])
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.ylim([0, 1])
        plt.grid()
    plt.plot([1, 0], [1, 1], 'k--', lw=1)
    plt.legend(loc="lower left", prop={"family":"meiryo"})
    plt.tight_layout()
    pr_filename = path.join(path.dirname(__file__), PR_FILENAME + ".pdf")
    plt.savefig(pr_filename)


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
    last_hidden_state, pooler_output = bert_model.bert(
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
        if ("label" in text) and (text["label"] in [0, 1, 2]):
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


# classifier.py [ID] [MODE]
# MODE:"r" or "f"
# ID: 227300(Eur truck sim), 255710(cities:Skylines)
if __name__ == '__main__':
    args = sys.argv
    if 3 <= len(args):
        if args[1].isdigit():
            ID = args[1]
            if args[2] in ["r", "f"]:
                MODE = "review" if args[2] == "r" else "forum"

                INPUT_REVIEW_FILENAME = "data/" + \
                    str(ID) + "/" + str(ID) + "_review_cleaned_out.json"
                INPUT_FORUM_FILENAME = "data/" + \
                    str(ID) + "/" + str(ID) + "_forum_cleaned.json"

                OUTPUT_FILENAME = "data/" + \
                    str(ID) + "/" + str(ID) + "_" + \
                    str(MODE) + "_predict" + ".json"
                MODEL_WEIGHT_FILENAME = "data/" + \
                    str(ID) + "/" + str(ID) + "_" + \
                    str(MODE) + "_model" + "/checkpoint"
                ROC_FILENAME = "data/" + \
                    str(ID) + "/" + str(ID) + "_" + str(MODE) + "_ROC"
                PR_FILENAME = "data/" + \
                    str(ID) + "/" + str(ID) + "_" + str(MODE) + "_PR"
                main()
            else:
                print('Argument must be "f" or "r"')
        else:
            print('Argument is not digit')
    else:
        print('Arguments are too short')
