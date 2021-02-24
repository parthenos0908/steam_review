import numpy as np
import pandas as pd
import csv
import os
from sklearn.feature_extraction.text import CountVectorizer

# stop_wordsはリストを指定．"english"を指定するといい感じに英語のいらん単語消してくれる
cv = CountVectorizer(stop_words="english")

docs = np.array(["Human machine interface for lab abc computer applications",
                 "A survey of user opinion of computer system response time",
                 "The EPS user interface management system",
                 "System and human system engineering testing of EPS",
                 "Relation of user perceived response time to error measurement",
                 "The generation of random binary unordered trees",
                 "The intersection graph of paths in trees",
                 "Graph minors IV Widths of trees and well quasi ordering",
                 "Graph minors A survey"])

# ベクトル化
vec = cv.fit_transform(docs)

# print(cv.vocabulary_)
# print(pd.DataFrame(vec.toarray(), columns=cv.get_feature_names()))

# VScodeの「ターミナルでpythonファイルを実行」から実行してもこのソースコードと同じディレクトリにcsvが保存されるよう設定
csv_path = os.path.join(os.path.dirname(__file__), 'BoW.csv')

with open(csv_path, 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(cv.get_feature_names())
    writer.writerows(vec.toarray())
