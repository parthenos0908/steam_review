import pandas as pd
import numpy as np
import torch
import csv
import os
import pprint
from transformers import BertTokenizer, BertModel

# Tokenize input
text = 'When people talk about Japan, they would always think about how innovative and technological this country gets! Or how pretty and neat the country is! Last but not the least, fashion, Cosplay and hype beast were always a big thing in the city of Japan. Coming to Japan with the intention of tourism would have been a great experience. Different culture. You can find a lot of unique things they sell in Japan! But as you live in Japan, you interact with the locals and everything doesn’t seem the way you thought of Japan.First thing I would like to discuss is how Japanese people are not flexible. They were taught to follow the rules and orders, which is good at some points but not so good at some points. There are always advantage and disadvantage. For example, when I crossed a small street, there were no cars at all, but people would choose to wait for the red lights to turn green. Since foreigners from all over the world start visiting Japan, Japanese people are much more open-minded. However, only certain elderly people would stare at you if you cross a small street with the red light. Not only about traffic light but customer services aren’t negotiable at times.  We, as a customer need a flexibility. For example, in some restaurants we have our own positions and we are not allowed to take others. It’s great that we could focus more on our own jobs, but as the customers are waiting in lines, we should help out our teammates, but some of the staffs wouldn’t care.Second, everybody who has their part-time jobs in a restaurant would have found out by now that Japanese restaurants love wasting all the clean leftover food. For instance, in Japan, they have different kinds of set meals for breakfast, lunch and dinner. As a restaurant, we need to prepare for breakfast, lunch and dinner set meals. If the meal we prepare for lunch hasn’t been sold out yet, the kitchen would definitely throw away all the leftover food after the lunch time ends. Why? Because they would like to have fresh food for the next day, not leftover food. It’s great but why not let us (staffs) eat the leftover stored in the refrigerator? I still can’t figure that out. It’s a waste of food. I know, Japan took really good care of the people so they won’t get food poisoning, but this is too much of a waste.'

# Load pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_text = tokenizer.tokenize(text)
tokenized_text.insert(0, "[CLS]")
tokenized_text.append("[SEP]")
# print(tokenized_text)

# テキストのままBERTに渡すのではなく，辞書で変換し，idになった状態にする
tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([tokens]).reshape(1, -1)
# print(tokens_tensor)

# BERTを読み込む．時間かかるかも
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

with torch.no_grad():
    all_encoder_layers = model(tokens_tensor, output_hidden_states=True)

# 0:last_hidden_​​state, 1:pooler_output, 2:hidden_​​states
print("Number of layers:", len(all_encoder_layers))                # 3
print("Number of batches:", len(all_encoder_layers[0]))            # 1
print("Number of tokens:", len(all_encoder_layers[0][0]))          # 506
print("Number of hidden units:", len(all_encoder_layers[0][0][0]))  # 768

# embedding = np.array(all_encoder_layers)
# print(embedding)

# VScodeの「ターミナルでpythonファイルを実行」から実行してもこのソースコードと同じディレクトリにcsvが保存されるよう設定
csv_path = os.path.join(os.path.dirname(__file__), 'BERT.csv')

# テキストのベクトル表現を保存
with open(csv_path, 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(all_encoder_layers[0][0][0].tolist())
