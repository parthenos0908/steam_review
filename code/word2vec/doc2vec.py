from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
from pprint import pprint
import nltk
# nltk.download('punkt')

docs = np.array(["Human machine interface for lab abc computer applications",
                 "Human machine interface for lab xyz computer applications",
                 "A survey of abc opinion of computer system response time",
                 "A survey of xyz opinion of computer system response time"])

# 単語に分割 + tag付け
corpus = [TaggedDocument(words=nltk.word_tokenize(doc), tags=[i])
          for i, doc in enumerate(docs)]

# モデル作成
# vector_size: 圧縮次元数
# min_count: 出現頻度の低いものをカットする
# window: 前後の単語を拾う際の窓の広さを決める
# epochs: 機械学習の繰り返し回数(デフォルト:10)十分学習できていないときにこの値を調整する
# epochs=10だと全然ダメ．1000にしたらちゃんと類似度検出できた
model = Doc2Vec(corpus, vector_size=50, window=5, min_count=1, epochs=1000)

pprint(model.wv.vocab)
pprint(model.docvecs.most_similar(0))
pprint(model.docvecs.most_similar(1))
pprint(model.docvecs.most_similar(2))
pprint(model.docvecs.most_similar(3))
