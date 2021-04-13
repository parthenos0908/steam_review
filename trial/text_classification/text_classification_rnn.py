import numpy as np
import tensorflow as tf
import transformers
from sklearn.metrics import accuracy_score

# model_nameはここから取得(cf. https://huggingface.co/transformers/pretrained_models.html)
# model_name = "cl-tohoku/bert-base-japanese"
model_name = "bert-base-uncased"
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

# 訓練データ
# train_texts = [
#     "この犬は可愛いです",
#     "その猫は気まぐれです",
#     "あの蛇は苦手です"
# ]
# train_labels = [1, 0, 0]  # 1: 好き, 0: 嫌い


train_texts = [
    "This dog is cute.",
    "That cat is fickle.",
    "I hate those snakes.",
    "you love me."
]
train_labels = [1, 0, 0, 1]  # 1: 好き, 0: 嫌い

# テストデータ
# test_texts = [
#     "その猫はかわいいです",
#     "どの鳥も嫌いです",
#     "あのヤギは怖いです"
# ]
# test_labels = [1, 0, 0]

test_texts = [
    "This dog is cute.",
    "I hate every bird.",
    "I'm afraid of that goat.",
    "I love tiger."
]
test_labels = [1, 0, 0, 1]

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


num_classes = 2
max_length = 15
batch_size = 10
epochs = 10

x_train = to_features(train_texts, max_length)
y_train = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
model = build_model(model_name, num_classes=num_classes, max_length=max_length)

# 訓練
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs
)

# 予測
x_test = to_features(test_texts, max_length)
y_test = np.asarray(test_labels)
y_preda = model.predict(x_test)
y_pred = np.argmax(y_preda, axis=1)
print(y_pred)
print("Accuracy: %.5f" % accuracy_score(y_test, y_pred))
