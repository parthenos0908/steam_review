import torch
from transformers import BertTokenizer, BertForMaskedLM

# Tokenize input
text = 'I would like to have a strong coffee.'

# Load pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_text = tokenizer.tokenize(text)
tokenized_text.insert(0, "[CLS]")
tokenized_text.append("[SEP]")
# ['[CLS]', 'i', 'would', 'like', 'to', 'have', 'a', 'strong', 'coffee', '.', '[SEP]']

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 3
tokenized_text[masked_index] = '[MASK]'
# ['[CLS]', 'i', '[MASK]', 'like', 'to', 'have', 'a', 'strong', 'coffee', '.', '[SEP]']

# テキストのままBERTに渡すのではなく，辞書で変換し，idになった状態にする
tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([tokens])
# BERTを読み込む．時間かかるかも
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]
# masked_indexとなっている部分の単語の予測結果を取り出し、その予測結果top5を出す
_, predict_indexes = torch.topk(predictions[0, masked_index], k=5)
predict_tokens = tokenizer.convert_ids_to_tokens(predict_indexes.tolist())
print(tokenized_text)
print(predict_tokens)
