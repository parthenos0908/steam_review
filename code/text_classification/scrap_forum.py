from bs4 import BeautifulSoup
from os import path
import urllib.request
import json
import time

# cities-skylinesのforumnoスクレイピング
steam_id = 255710
paradox_url = "https://forum.paradoxplaza.com"

def main():
    general_url = "/forum/forums/cities-skylines.859/"
    general_forum = loadForums(general_url, "General", page_MAX = 283) # 283
    del general_forum[0:5] # Sticky threads の除去

    bug_url = "/forum/forums/support-bug-reports.879/"
    bug_forum = loadForums(bug_url, "Bug", page_MAX = 350) # 350
    del bug_forum[0:5] # Sticky threads の除去

    feature_url = "/forum/forums/suggestions-feedback.881/"
    feature_forum = loadForums(feature_url, "Feature", page_MAX = 141) # 141
    del feature_forum[0:2] # Sticky threads の除去

    forum_list = general_forum + bug_forum + feature_forum

    print(len(forum_list))

    json_filename = path.join(path.dirname(__file__), "forumData/" + str(steam_id) + "_forum.json")
    with open(json_filename, mode='w') as f:
        json.dump(forum_list, f, sort_keys=True, indent=4)

def loadForums(url, label, page_MAX):
    forum_list = []
    count = 0
    for i in range(page_MAX):
        html = urllib.request.urlopen(paradox_url + url + "page-" + str(i+1))
        soup = BeautifulSoup(html, "html.parser")
        threads = soup.find_all(class_="structItem-title")

        for thread in threads:
            count += 1
            print("{0}:{1}".format(label, count))
            titel = thread.get_text(strip=True)
            if not titel:
                titel = ""
            thread_url = thread.contents[1].get("data-preview-url") # /forum/threads/[title]/preview
            if thread_url:
                comment = loadThread(paradox_url + thread_url)
            else:
                comment = ""
            forum = {
                "title":titel,
                "comment":comment,
                "label":label
            }
            forum_list.append(forum)
            time.sleep(0.01)
    
    return forum_list

def loadThread(url):
    try:
        html = urllib.request.urlopen(url)
        soup = BeautifulSoup(html, "html.parser")
        comment = soup.find(class_="bbWrapper").get_text(strip=True)
        return comment
    except Exception as e:
        print(e)
        return ""

if __name__ == '__main__':
    main()