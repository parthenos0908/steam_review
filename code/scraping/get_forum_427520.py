from bs4 import BeautifulSoup
from os import path
import urllib.request
import json
import time
import re
from tqdm import tqdm # 進捗表示

from nltk.util import pr

# Factorioのforumnoスクレイピング
steam_id = 427520
factorio_url = "https://forums.factorio.com/"


def main():
    general_url = "viewforum.php?f=5"
    general_forum = loadForums(general_url, "General", page_MAX=134) #134

    bug_url = "viewforum.php?f=11"
    bug_forum = loadForums(bug_url, "Bug", page_MAX=360) #360

    feature_url = "viewforum.php?f=6"
    feature_forum = loadForums(feature_url, "Feature", page_MAX=341) #341

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
        html = urllib.request.urlopen(factorio_url + url + "&start=" + str(i*25))
        soup = BeautifulSoup(html, "html.parser")
        threads = soup.find_all(class_="topictitle")

        for i, thread in enumerate(tqdm(threads, leave=False)):
        # for i, thread in enumerate(threads):
            if (label == "General" and i == 0):
                continue
            if (label == "Feature" and i in [0,1,2,3]):
                continue
            title = thread.get_text()
            if not title:
                title = ""
            thread_url = thread.get("href")  # /forum/threads/[title]/preview
            if thread_url:
                forum_url = factorio_url + thread_url[2:] # ./viewtipic~ となっているので前2文字を削除して結合
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
        comment = soup.find(class_="content").get_text()
        return comment
    except Exception as e:
        print(e)
        return ""


if __name__ == '__main__':
    # loadForums("viewforum.php?f=6", "Feature", 1)
    main()