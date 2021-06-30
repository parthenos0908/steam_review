import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import json
from os import path
from googletrans import Translator

translator = Translator()

INPUT_FILENAME = "test_in.json"
OUTPUT_FILENAME = "test_out.json"

WIDTH = 1000
HEIGHT = 700

LABEL = {
    0: "Bug Report",
    1: "Feature Request",
    2: "Other",
    3: "-"
}


def main():
    # アプリの実行
    root = tk.Tk()
    app = labelingApp(master=root)
    app.pack(fill=tk.Y, expand=True)
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

        # labelの表示
        self.review_label = tk.Label(self, text='review',
                                     font=('', 20, 'bold'),
                                     foreground='#ffffff',
                                     background='#0000aa')
        self.review_label.place(x=WIDTH/100, y=HEIGHT/20,
                                width=100, height=30)
        self.translated_review_label = tk.Label(self, text='review(google翻訳)',
                                                font=('', 20, 'bold'),
                                                foreground='#ffffff',
                                                background='#aa00aa')
        self.translated_review_label.place(
            x=2*WIDTH/5+WIDTH/100, y=HEIGHT/20, width=275, height=30)

        # 複数行のテキストフィールドの生成
        self.review_field = ScrolledText()
        self.review_field.configure(font=("Calibri", 16, "normal"))
        self.review_field.place(x=WIDTH/100, y=HEIGHT/10,
                                width=2*WIDTH/5-2*(WIDTH/100), height=6*HEIGHT/10)
        self.translated_review_field = ScrolledText()
        self.translated_review_field.configure(font=("Calibri", 16, "normal"))
        self.translated_review_field.place(
            x=2*WIDTH/5+WIDTH/100, y=HEIGHT/10, width=2*WIDTH/5-2*(WIDTH/100), height=6*HEIGHT/10)

        # チェックボックスの生成（翻訳機能のオンオフ）
        self.isTrans = tk.BooleanVar()
        self.isTrans.set(True)
        self.checkbox = ttk.Checkbutton(self,
                                        variable=self.isTrans, text='翻訳機能')
        self.checkbox.place(x=2*WIDTH/5+WIDTH/100, y=7*HEIGHT /
                            10+WIDTH/100, width=2*WIDTH/5-2*(WIDTH/100), height=30)

        # リストボックスの生成
        items = []
        for i, data in enumerate(self.input_data):
            items.append("{0} : {1}".format(
                str(i).rjust(8, " "), data["label"]))
            if i > 100:
                break
        self.list_box = tk.Listbox(self, listvariable=tk.StringVar(
            value=items), selectmode='browse')
        self.list_box.bind('<<ListboxSelect>>', lambda e: self.on_select())
        self.list_box.place(x=4*WIDTH/5+WIDTH/100, y=HEIGHT/20,
                            width=WIDTH/5-4*(WIDTH/100), height=18*HEIGHT/20)
        self.scrollbar = ttk.Scrollbar(
            self, orient=tk.VERTICAL, command=self.list_box.yview)
        self.scrollbar.place(x=WIDTH-3*(WIDTH/100), y=HEIGHT/20,
                             width=2*WIDTH/100, height=18*HEIGHT/20)
        self.list_box['yscrollcommand'] = self.scrollbar.set

        # ラジオボタンの値
        self.tag_value = tk.IntVar()
        self.tag_value.set(2)

        # ラジオボタンの生成（タグ付け用）
        text_height = 30
        self.radiobutton1 = ttk.Radiobutton(
            self, variable=self.tag_value, value=0, text='バグ報告')
        self.radiobutton1.place(
            x=WIDTH/100, y=7*HEIGHT/10+5, width=100, height=text_height)
        self.radiobutton2 = ttk.Radiobutton(
            self, variable=self.tag_value, value=1, text='機能要求')
        self.radiobutton2.place(
            x=WIDTH/100, y=7*HEIGHT/10+5+(5+text_height), width=100, height=text_height)
        self.radiobutton3 = ttk.Radiobutton(
            self, variable=self.tag_value, value=2, text='その他')
        self.radiobutton3.place(
            x=WIDTH/100, y=7*HEIGHT/10+5+2*(5+text_height), width=100, height=text_height)
        self.radiobutton4 = ttk.Radiobutton(
            self, variable=self.tag_value, value=3, text='未定義')
        self.radiobutton4.place(
            x=WIDTH/100, y=7*HEIGHT/10+5+3*(5+text_height), width=100, height=text_height)

        # NEXTボタンの生成
        self.nextButton = tk.Button(
            self, text='Next >', command=self.on_click_next)
        self.nextButton.place(x=WIDTH/100+WIDTH/10+WIDTH/100, y=19*HEIGHT/20-5,
                              width=WIDTH/10, height=HEIGHT/20)

        # BACKボタンの生成
        self.backButton = tk.Button(
            self, text='< Back', command=self.on_click_back)
        self.backButton.place(x=WIDTH/100, y=19*HEIGHT/20-5,
                              width=WIDTH/10, height=HEIGHT/20)

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
