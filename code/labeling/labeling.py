import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import json
from os import path
from typing import Text
from googletrans import Translator

translator = Translator()

INPUT_FILENAME = "test_in.json"
OUTPUT_FILENAME = "test_out.json"

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
    app.bind("<Configure>", app.change_size)
    root.bind("<Key>", app.on_press_key)
    app.mainloop()


class labelingApp(tk.Frame):
    # 初期化
    def __init__(self, master=None):
        tk.Frame.__init__(self, master, width=WIDTH, height=HEIGHT)

        # iteratorの初期化
        self.iterator = 0

        input_filepath = path.join(path.dirname(__file__), INPUT_FILENAME)
        self.input_data = load_json(input_filepath)

        # タイトルの表示
        self.master.title('review labeling')

        text_box = ttk.Frame(self, relief=tk.RIDGE)
        text_box.place(x=2, rely=0, relwidth=4/5, relheight=3/4)

        origin = ttk.Frame(text_box, relief=tk.RIDGE)
        origin.place(relx=0, rely=0, relwidth=1/2, relheight=1)
        translated = ttk.Frame(text_box, relief=tk.RIDGE)
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
        scroll_list_box = tk.Frame(self, relief=tk.RIDGE)
        scroll_list_box.place(
            relx=1, rely=0, relwidth=1/5, relheight=99/100, anchor=tk.NE)
        items = []
        for i, data in enumerate(self.input_data):
            items.append("{0} : {1}".format(
                str(i).rjust(8, " "), data["label"]))
            if i > 100:
                break
        self.list_box = tk.Listbox(scroll_list_box, listvariable=tk.StringVar(
            value=items), selectmode='browse')
        self.list_box.bind('<<ListboxSelect>>', lambda e: self.on_select())
        self.list_box.place(relx=0, rely=0, relwidth=9/10, relheight=1)
        self.scrollbar = ttk.Scrollbar(
            scroll_list_box, orient=tk.VERTICAL, command=self.list_box.yview)
        self.scrollbar.place(relx=9/10, rely=0, relwidth=1/10, relheight=1)
        self.list_box['yscrollcommand'] = self.scrollbar.set

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
        button.place(x=110, rely=99/100,  width=200, height=50, anchor=tk.SW)

        # BACKボタンの生成
        self.backButton = tk.Button(
            button, text='< Back', command=self.on_click_back)
        self.backButton.pack(expand=True, side=tk.LEFT, padx=10, fill=tk.BOTH)

        # NEXTボタンの生成
        self.nextButton = tk.Button(
            button, text='Next >', command=self.on_click_next)
        self.nextButton.pack(expand=True, side=tk.LEFT, padx=10, fill=tk.BOTH)

        # 初期化時実行関数
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
        if key == "Right":
            self.on_click_next()
        elif key == "Left":
            self.on_click_back()
        elif key == "Up":
            i = self.tag_value.get()
            if i != 0:
                self.tag_value.set(i-1)
        elif key == "Down":
            i = self.tag_value.get()
            if i != len(LABEL)-1:
                self.tag_value.set(i+1)

    def on_click_next(self):
        self.add_tag()
        self.iterator += 1
        self.display_review()

    def on_click_back(self):
        self.iterator -= 1 if self.iterator != 0 else 0
        self.display_review()

    def on_select(self):
        # curselectionの返り値はtuple
        self.iterator = self.list_box.curselection()[0]
        self.display_review()

    def display_review(self):
        self.review = self.input_data[self.iterator]["review"]
        if (self.review == "") or (self.isTrans.get() == False):
            self.review == ""
            self.translated_review = ""
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

    def add_tag(self):
        self.input_data[self.iterator]["label"] = self.tag_value.get()
        self.list_box.insert(self.iterator, "{0} : {1}".format(
            str(self.iterator).rjust(8, " "), LABEL[self.input_data[self.iterator]["label"]]))
        if self.input_data[self.iterator]["label"] == 0:
            text_color = "red"
        elif self.input_data[self.iterator]["label"] == 1:
            text_color = "green"
        elif self.input_data[self.iterator]["label"] == 2:
            text_color = "blue"
        elif self.input_data[self.iterator]["label"] == 3:
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
    main()
