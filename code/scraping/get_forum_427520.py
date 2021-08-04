from bs4 import BeautifulSoup
from os import path
import urllib.request
import json
import time
import re

from nltk.util import pr

# Factorioのforumnoスクレイピング
steam_id = 427520
factorio_url = "https://forums.factorio.com/"


def main():
    general_url = "viewforum.php?f=5"
    general_forum = loadForums(general_url, "General", page_MAX=358)

    bug_url = "viewforum.php?f=11"
    bug_forum = loadForums(bug_url, "Bug", page_MAX=351)

    feature_url = "viewforum.php?f=6"
    feature_forum = loadForums(feature_url, "Feature", page_MAX=141)

    forum_list = general_forum + bug_forum + feature_forum

    print(len(forum_list))

    json_filename = path.join(path.dirname(
        __file__), "forumData/" + str(steam_id) + "_forum.json")
    with open(json_filename, mode='w') as f:
        json.dump(forum_list, f, sort_keys=True, indent=4)


def loadForums(url, label, page_MAX):
    forum_list = []
    count = 0
    for i in range(page_MAX):
        html = urllib.request.urlopen(factorio_url + url + "&start=" + str(i*25))
        soup = BeautifulSoup(html, "html.parser")
        threads = soup.find_all(class_="topictitle")

        for thread in threads:
            print(thread)
            count += 1
            # print("{0}:{1}".format(label, count),end="")
            # title = thread.get_text()
            # if not title:
            #     title = ""
            # elif title.startswith("\nCities: Skylines (Steam) - "): # formatが決まっているバグ報告の例外処理
            #     title = title.replace("\nCities: Skylines (Steam) - ", "")
            # thread_url = thread.contents[1].get(
            #     "data-preview-url")  # /forum/threads/[title]/preview
            # if thread_url:
            #     forum_url = paradox_url + thread_url
            #     comment = loadThread(forum_url)
            # else:
            #     print(" - non comment")
            #     forum_url = ""
            #     comment = ""
            # forum = {
            #     "title": title,
            #     "comment": comment,
            #     "label": label,
            #     "url": forum_url
            # }
            # forum_list.append(forum)
            # time.sleep(0.01)
        print(count)


    return forum_list


def loadThread(url):
    try:
        html = urllib.request.urlopen(url)
        soup = BeautifulSoup(html, "html.parser")
        comment = soup.find(class_="bbWrapper").get_text()
        format_comment = re.search(r'Please explain your issue is in as much detail as possible\.(\r\n|\n|\r|.)*Can you replicate the issue\? If yes, please explain how you did it\.', comment)
        if format_comment is not None:
            print(" -format")
            comment = format_comment.group().replace("Please explain your issue is in as much detail as possible.", "").replace("Can you replicate the issue? If yes, please explain how you did it.", "")
        else: print("")
        return comment
    except Exception as e:
        print(e)
        return ""


if __name__ == '__main__':
    loadForums("viewforum.php?f=5", "a", 1)
    # main()