import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

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

print(cv.vocabulary_)
print(pd.DataFrame(vec.toarray(), columns=cv.get_feature_names()))