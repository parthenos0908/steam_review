# DLパラメータ
num_classes = 3
max_length = 128  # 256はメモリ不足
batch_size = 32
max_epochs = 10
hold_out_rate = 0.7  # 訓練データとテストデータの比率

# 設定
is_learn = True  # 新しく学習する(True) or 既存の学習結果使う(False)
is_del_less_words = True  # 1単語以下をテストデータから除外
is_less_train_data = False  # フォーラムの教師データ数をレビューに合わせる
is_earlystop = True  # 早期終了を行う
