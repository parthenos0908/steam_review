import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import json
import random
from os import path
from typing import Text
from googletrans import Translator
import time

translator = Translator()

# INPUT_FILENAME = "255710_review_cleaned_5000_out.json"
# OUTPUT_FILENAME = "255710_review_cleaned_5000_out.json"

INPUT_FILENAME = "review/255710_review_BR_filtered_3757_out.json"
OUTPUT_FILENAME = "review/255710_review_BR_filtered_3757_out.json"

WIDTH = 1000
HEIGHT = 700

MIN_WIDTH = 660
MIN_HEIGHT = 520

LABEL = {
    0: "Bug Report",
    1: "Feature Request",
    2: "Other",
    3: "-"
}


def main():
    # アプリの実行
    root = tk.Tk()
    root.minsize(MIN_WIDTH, MIN_HEIGHT)
    app = labelingApp(master=root)
    app.pack(fill=tk.BOTH, expand=True)
    # app.bind("<Configure>", app.change_size)
    root.bind("<Key>", app.on_press_key)
    app.mainloop()


class labelingApp(tk.Frame):
    # 初期化
    def __init__(self, master=None):
        tk.Frame.__init__(self, master, width=WIDTH, height=HEIGHT)

        # iteratorの初期化
        self.iterator = 0

        input_filepath = path.join(path.dirname(__file__), INPUT_FILENAME)
        self.json_data = load_json(input_filepath)
        self.data_size = len(self.json_data)

        # タイトルの表示
        self.master.title(
            'review labeling ({0}/{1})'.format(self.iterator, len(self.json_data)))

        text_box = ttk.Frame(self, relief=tk.FLAT)
        text_box.place(x=2, y=1, relwidth=4/5, relheight=4/5)

        origin = ttk.Frame(text_box, relief=tk.FLAT)
        origin.place(relx=0, rely=0, relwidth=1/2, relheight=1)
        translated = ttk.Frame(text_box, relief=tk.FLAT)
        translated.place(relx=1/2, rely=0, relwidth=1/2, relheight=1)

        # labelの表示
        self.review_label = tk.Label(origin, text='review',
                                     font=('', 20, 'bold'),
                                     foreground='#ffffff',
                                     background='#0000aa')
        self.review_label.pack(fill=tk.BOTH, pady=1)
        self.translated_review_label = tk.Label(translated, text='review(google翻訳)',
                                                font=('', 20, 'bold'),
                                                foreground='#ffffff',
                                                background='#aa00aa')
        self.translated_review_label.pack(fill=tk.BOTH, pady=1)

        # 複数行のテキストフィールドの生成
        self.review_field = ScrolledText(origin, wrap=tk.WORD)
        self.review_field.configure(font=("Calibri", 16, "normal"))
        self.review_field.pack()
        self.translated_review_field = ScrolledText(translated, wrap=tk.WORD)
        self.translated_review_field.configure(font=("Calibri", 16, "normal"))
        self.translated_review_field.pack()

        # チェックボックスの生成（翻訳機能のオンオフ）
        self.isTrans = tk.BooleanVar()
        self.isTrans.set(True)
        self.checkbox = ttk.Checkbutton(self,
                                        variable=self.isTrans, text='翻訳機能')
        self.checkbox.place(relx=4/5, rely=99/100,  width=80,
                            height=30, anchor=tk.SE)

        # リストボックスの生成
        scroll_list_box = tk.Frame(self, relief=tk.FLAT)
        scroll_list_box.place(
            relx=1, y=1, relwidth=1/5, relheight=99/100, anchor=tk.NE)
        items = []

        self.list_box = tk.Listbox(scroll_list_box, listvariable=tk.StringVar(
            value=items), selectmode='browse')
        self.list_box.bind('<<ListboxSelect>>', lambda e: self.on_select())
        self.list_box.place(relx=0, rely=0, relwidth=9/10, relheight=1)
        self.scrollbar = ttk.Scrollbar(
            scroll_list_box, orient=tk.VERTICAL, command=self.list_box.yview)
        self.scrollbar.place(relx=9/10, rely=0, relwidth=1/10, relheight=1)
        self.list_box['yscrollcommand'] = self.scrollbar.set

        # リストボックスの初期状態を描画（入力データのラベル情報を表示）
        count_bug = 0
        count_feature = 0
        for i, data in enumerate(self.json_data):
            if data.get('label') in (0, 1, 2, 3):
                self.list_box.insert(i, "{0} : {1}".format(
                    str(i).rjust(8, " "), LABEL[data["label"]]))
                if data["label"] == 0:
                    text_color = "red"
                    count_bug += 1
                elif data["label"] == 1:
                    text_color = "green"
                    count_feature += 1
                elif data["label"] == 2:
                    text_color = "blue"
                else:
                    text_color = "black"
            else:
                self.list_box.insert(i, "{0} : {1}".format(
                    str(i).rjust(8, " "), LABEL[3]))
                text_color = "black"
            self.list_box.itemconfig(i, foreground=text_color)
            # if i == 100:
            #     break
        print("bug:{0}, feature:{1}".format(count_bug, count_feature))

        # ラジオボタンの親frame
        radio = ttk.Frame(self, relief=tk.RIDGE)
        radio.place(x=2, rely=99/100, width=100,
                    height=120, anchor=tk.SW)

        # ラジオボタンの値
        self.tag_value = tk.IntVar()
        self.tag_value.set(2)

        # ラジオボタンの生成（タグ付け用）
        self.radiobutton1 = ttk.Radiobutton(
            radio, variable=self.tag_value, value=0, text='バグ報告')
        self.radiobutton1.pack(expand=True, anchor=tk.W, padx=10)
        self.radiobutton2 = ttk.Radiobutton(
            radio, variable=self.tag_value, value=1, text='機能要求')
        self.radiobutton2.pack(expand=True, anchor=tk.W, padx=10)
        self.radiobutton3 = ttk.Radiobutton(
            radio, variable=self.tag_value, value=2, text='その他')
        self.radiobutton3.pack(expand=True, anchor=tk.W, padx=10)
        self.radiobutton4 = ttk.Radiobutton(
            radio, variable=self.tag_value, value=3, text='未定義')
        self.radiobutton4.pack(expand=True, anchor=tk.W, padx=10)

        # ボタンの親frame
        button = ttk.Frame(self, relief=tk.FLAT)
        button.place(x=110, rely=99/100,  width=300, height=50, anchor=tk.SW)

        # BACKボタンの生成
        self.backButton = tk.Button(
            button, text='< Back', command=self.on_click_back)
        self.backButton.pack(expand=True, side=tk.LEFT, padx=10, fill=tk.BOTH)

        # NEXTボタンの生成
        self.nextButton = tk.Button(
            button, text='Next >', command=self.on_click_next)
        self.nextButton.pack(expand=True, side=tk.LEFT, padx=10, fill=tk.BOTH)

        # SAVEボタンの生成
        self.nextButton = tk.Button(
            button, text='[SAVE]', command=self.on_click_save, relief=tk.SOLID)
        self.nextButton.pack(expand=True, side=tk.LEFT, padx=10, fill=tk.BOTH)

        # 初期化時実行関数
        self.iterator_next = -1
        self.translated_review_next = ""
        self.display_review()

    def change_size(self, e):
        global WIDTH, HEIGHT
        WIDTH = e.width + e.x
        HEIGHT = e.height + e.y
        print(e)
        print("{0}:{1}".format(WIDTH, HEIGHT))

    def on_press_key(self, e):
        key = e.keysym
        # print(key)
        if key == "f":
            self.on_click_next()
        elif key == "s":
            self.on_click_back()
        elif key == "e":
            i = self.tag_value.get()
            if i != 0:
                self.tag_value.set(i-1)
        elif key == "d":
            i = self.tag_value.get()
            if i != len(LABEL)-1:
                self.tag_value.set(i+1)

    def on_click_next(self):
        self.add_tag()
        self.iterator += 1
        self.display_review()
        self.after(0, self.pre_load_next_trans)
        self.iterator_next = self.iterator + 1

    def on_click_back(self):
        self.iterator -= 1 if self.iterator != 0 else 0
        tmp = self.translated_review
        self.display_review()
        self.translated_review_next = tmp
        self.iterator_next = self.iterator + 1

    def on_click_save(self):
        output_filepath = path.join(path.dirname(__file__), OUTPUT_FILENAME)
        save_json(self.json_data, output_filepath)

    def on_select(self):
        # curselectionの返り値はtuple
        self.iterator = self.list_box.curselection()[0]
        self.display_review()

    def display_review(self):
        self.review = self.json_data[self.iterator]["review"]
        if (self.review == "") or (self.isTrans.get() == False):
            self.review == ""
            self.translated_review = ""
        elif self.iterator_next == self.iterator:
            self.translated_review = self.translated_review_next
        else:
            self.translated_review = translator.translate(
                self.review, src="en", dest="ja").text
        # reviewの表示
        self.review_field.configure(stat="normal")
        self.review_field.delete('1.0', 'end')
        self.review_field.insert('1.0', self.review)
        self.review_field.configure(stat="disable")
        # review(翻訳)の表示
        self.translated_review_field.configure(stat="normal")
        self.translated_review_field.delete('1.0', 'end')
        self.translated_review_field.insert('1.0', self.translated_review)
        self.translated_review_field.configure(stat="disable")
        # タイトルの変更
        self.master.title(
            'review labeling ({0}/{1})'.format(self.iterator, self.data_size))

    # 次の文の英訳を先に読み込み
    def pre_load_next_trans(self):
        self.translated_review_next = translator.translate(
            self.json_data[self.iterator + 1]["review"], src="en", dest="ja").text

    def add_tag(self):
        self.json_data[self.iterator]["label"] = self.tag_value.get()
        self.list_box.insert(self.iterator, "{0} : {1}".format(
            str(self.iterator).rjust(8, " "), LABEL[self.json_data[self.iterator]["label"]]))
        self.list_box.see(self.iterator)
        if self.json_data[self.iterator]["label"] == 0:
            text_color = "red"
        elif self.json_data[self.iterator]["label"] == 1:
            text_color = "green"
        elif self.json_data[self.iterator]["label"] == 2:
            text_color = "blue"
        elif self.json_data[self.iterator]["label"] == 3:
            text_color = "black"
        self.list_box.itemconfig(self.iterator, foreground=text_color)
        self.list_box.delete(self.iterator+1)


def load_json(json_filepath):
    with open(json_filepath, mode='r') as f:
        json_data = json.load(f)
    return json_data


def save_json(output_list, json_filepath):
    with open(json_filepath, mode='w') as f:
        json.dump(output_list, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    # input_filepath = path.join(path.dirname(__file__), INPUT_FILENAME)
    # json_data = load_json(input_filepath)
    # out_data = random.sample(json_data, 5000)
    # output_filepath = path.join(path.dirname(
    #     __file__), "255710_review_cleaned_5000.json")
    # save_json(out_data, output_filepath)
    main()
