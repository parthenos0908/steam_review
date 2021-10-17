from bs4 import BeautifulSoup
from os import path
import urllib.request
import json
import time
import re
import math
from bs4.element import Comment
from tqdm import tqdm  # 進捗表示

from nltk.util import pr

import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # SSL証明


# Euro Truck Simulator 2のforumスクレイピング
steam_id = 227300
euro_url = "https://forum.scssoft.com/"

# BRのforumの(url番号, topic数)
BR_LINK = [(113, 105), (124, 92), (125, 42), (126, 12), (127, 16), (128, 36), (129, 124), (130, 130), (131, 88), (132, 17), (133, 23), (136, 69), (140, 29), (141, 8), (142, 76), (143, 249), (146, 65), (147, 131), (148, 63), (149, 29), (151, 191), (153, 91), (157, 73), (166, 253), (167, 129),
           (168, 244), (170, 96), (171, 74), (173, 39), (174, 157), (209, 71), (215, 146), (218, 212), (220, 378), (225, 283), (228, 349), (230, 982), (234, 548), (235, 1470), (238, 642), (261, 450), (264, 899), (268, 603), (272, 495), (275, 466), (277, 380), (278, 1549), (283, 370), (288, 60)]


def main():
    general_url = "viewforum.php?f=41"
    general_forum = loadForums(general_url, "General", page_MAX=168)  # 168

    # BRはver毎に1forumが用意されている
    bug_forum = []
    for url_num, topics in BR_LINK:
        bug_url = "viewforum.php?f=" + str(url_num)
        page = math.ceil(topics/float(25))
        bug_forum.extend(loadForums(bug_url, "Bug", page_MAX=page))

    feature_url = "viewforum.php?f=5"
    feature_forum = loadForums(feature_url, "Feature", page_MAX=192)  # 192

    forum_list = general_forum + bug_forum + feature_forum

    print(len(forum_list))

    json_filename = path.join(path.dirname(
        __file__), "forumData/" + str(steam_id) + "_forum.json")
    with open(json_filename, mode='w') as f:
        json.dump(forum_list, f, sort_keys=True, indent=4)


def loadForums(url, label, page_MAX):
    forum_list = []
    count = 0
    for i in tqdm(range(page_MAX)):
        # for i in range(page_MAX):
        html = urllib.request.urlopen(euro_url + url + "&start=" + str(i*25))
        soup = BeautifulSoup(html, "html.parser")
        threads = soup.find_all(class_="topictitle")

        for i, thread in enumerate(tqdm(threads, leave=False)):
            # for i, thread in enumerate(threads):
            if (label == "General" and i == 0):
                continue
            if (label == "Bug" and i == 0):
                continue
            if (label == "Feature" and i == 0):
                continue
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
            forum_list.append(forum)
            time.sleep(0.01)

    return forum_list


def loadThread(url):
    try:
        html = urllib.request.urlopen(url)
        soup = BeautifulSoup(html, "html.parser")
        first_post = soup.find(class_="postbody")
        comment = first_post.find(class_="content").get_text()
        return comment
    except Exception as e:
        print(e)
        return ""


if __name__ == '__main__':
    main()
