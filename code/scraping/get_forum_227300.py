from bs4 import BeautifulSoup
from os import error, path
import urllib.request
import json
import time
import re
import math
from bs4.element import Comment

# 進捗表示
from tqdm import tqdm
from collections import OrderedDict

# マルチスレッド
from concurrent.futures import ThreadPoolExecutor

from nltk.util import pr

import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # SSL証明


# Euro Truck Simulator 2のforumスクレイピング
steam_id = 227300
euro_url = "https://forum.scssoft.com/"

# BRのforumの(url番号, topic数) 49個
BR_LINK = [(113, 105), (124, 92), (125, 42), (126, 12), (127, 16), (128, 36), (129, 124), (130, 130), (131, 88), (132, 17), (133, 23), (136, 69), (140, 29), (141, 8), (142, 76), (143, 249), (146, 65), (147, 131), (148, 63), (149, 29), (151, 191), (153, 91), (157, 73), (166, 253), (167, 129),
           (168, 244), (170, 96), (171, 74), (173, 39), (174, 157), (209, 71), (215, 146), (218, 212), (220, 378), (225, 283), (228, 349), (230, 982), (234, 548), (235, 1470), (238, 642), (261, 450), (264, 899), (268, 603), (272, 495), (275, 466), (277, 380), (278, 1549), (283, 370), (288, 60)]


def main():
    json_filename = path.join(path.dirname(
        __file__), "forumData/" + str(steam_id) + "_forum.json")
    forum_list = []

    general_url = "viewforum.php?f=41"
    general_forum = loadForums(general_url, "General", page_MAX=168)  # 168
    forum_list.extend(general_forum)
    save_json(forum_list, json_filename)

    # BRはver毎に1forumが用意されている
    bug_forum = []
    BR_count = 0
    for url_num, topics in BR_LINK:
        bug_url = "viewforum.php?f=" + str(url_num)
        page = math.ceil(topics/float(25))
        bug_forum.extend(loadForums(bug_url, "Bug", page_MAX=page, BR_num = BR_count))
        BR_count += 1
    forum_list.extend(bug_forum)
    save_json(forum_list, json_filename)

    feature_url = "viewforum.php?f=5"
    feature_forum = loadForums(feature_url, "Feature", page_MAX=192)  # 192
    forum_list.extend(feature_forum)
    save_json(forum_list, json_filename)

    print(len(forum_list))

def loadForums(url, label, page_MAX, BR_num = -1):
    forum_list = []
    count = 0
    # for i in range(page_MAX):
    with tqdm(range(page_MAX)) as pbar1:
        pbar1.set_description("[" + label + (str(BR_num) if BR_num != -1 else "") + "]")
        for i in pbar1:
            html = urllib.request.urlopen(euro_url + url + "&start=" + str(i*25))
            soup = BeautifulSoup(html, "html.parser")
            threads = soup.find_all(class_="topictitle") #26件
            threads.pop(0) # EurではForumの１つ目のみが常に固定トピック
            CPU_core = 25
            for _ in range(len(threads)):
                labellist = []
            for _ in range(CPU_core):
                if len(threads):
                    labellist.append(label)
            with ThreadPoolExecutor(CPU_core) as executor:
                results = list(executor.map(getForum, threads, labellist))
            forum_list.extend(results)

    return forum_list

def getForum(thread, label):
    title = thread.get_text()
    if not title:
        title = ""
    thread_url = thread.get("href")
    if thread_url:
        # ./viewtipic~ となっているので前2文字を削除して結合
        forum_url = euro_url + thread_url[2:]
        comment = loadThread(forum_url)
    else:
        print(" - non comment")
        forum_url = ""
        comment = ""
    forum = {
        "title": title,
        "comment": comment,
        "label": label,
        "url": forum_url
    }
    return forum

def loadThread(url):
    try:
        html = urllib.request.urlopen(url)
        soup = BeautifulSoup(html, "html.parser")
        first_post = soup.find(class_="postbody")
        comment = first_post.find(class_="content").get_text()
        return comment
    except Exception as e:
        print(e)

        error_log = path.join(path.dirname(__file__), "forumData/" + str(steam_id) + "_error.txt")
        with open(error_log, mode='a') as f:
            print("[" + e + "]" + url)

        return ""

def save_json(output_list, json_filepath):
    with open(json_filepath, mode='w') as f:
        json.dump(output_list, f, sort_keys=True, indent=4)

if __name__ == '__main__':
    main()
