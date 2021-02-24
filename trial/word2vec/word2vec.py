from gensim.models import word2vec
import numpy as np
from pprint import pprint
import nltk
# nltk.download('punkt')

docs = np.array(["Human machine interface for lab abc computer applications",
                 "A survey of user opinion of computer system response time",
                 "The EPS user interface management system",
                 "System and human system engineering testing of EPS",
                 "Relation of user perceived response time to error measurement",
                 "The generation of random binary unordered trees",
                 "The intersection graph of paths in trees",
                 "Graph minors IV Widths of trees and well quasi ordering",
                 "Graph minors A survey"])

# 単語に分割
corpus = [nltk.word_tokenize(doc) for doc in docs]

# モデル作成
# size: 圧縮次元数
# min_count: 出現頻度の低いものをカットする
# window: 前後の単語を拾う際の窓の広さを決める
# iter: 機械学習の繰り返し回数(デフォルト:5)十分学習できていないときにこの値を調整する
# model.wv.most_similarの結果が1に近いものばかりで、model.dict['wv']のベクトル値が小さい値ばかりのときは、学習回数が少ないと考えられます。その場合、iterの値を大きくして、再度学習を行います。
model = word2vec.Word2Vec(corpus, size=100, min_count=1, window=3, iter=100)

print(model.wv.vocab)
print(model.__dict__['wv']['Graph'])
print(model.wv.similarity(w1="computer", w2="trees"))
