#import moesearch
import json
from os import error, wait
import requests
import re
import time
import sys

boards_list = [
    "3", "a", "adv", "an", "asp", "b",  "biz",
     "cgl", "ck",  "co", "diy",
    "fa", "fit", "g", "gd", "his",
    "ic", "jp", "k", "lgbt", "lit", "m",
    "mlp", "mu", "n", "news", "o", "out", "p", "po", "pol",
    "pw", "qst", "r9k", "s4s","sci", "soc", "sp",
    "tg", "toy", "trv", "tv", "v", "vg", "vm", "vmg",
    "vr", "vrpg", "vst", "vp", "vt", "w", "wg", "wsg", "wsr",
    "x", "xs"]

four_plebs = ['adv', 'tv', 'x', 'pol', 's4s', 'tg', 'trv', 'sp']
archived_moe = ['','','']
desuarchive = ['a', 'an', 'cgl', 'co', 'fit', 'g', 'his', 'k', 'm', 'mlp', 'mu', 'r9k', 'vr']
selected = four_plebs
website = sys.argv[1]
if website == '4plebs':
    website = "https://archive.4plebs.org/"
    selected = four_plebs
elif website == "archived":
    website = "https://archived.moe/"
    selected = archived_moe
elif website == "desu":
    website = "https://desuarchive.org/"
    selected = desuarchive
else:
    print("please parse a website: 4plebs, archived, desu")

first_post = 12642614 
current_post = first_post

def get_thread(website = website, board='a', post='0'):
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'pleaseLet me Use this thing'

    }
    
    params = {
        'board': board,
        'num': post,
    }
    response = requests.get(website + '_/api/chan/thread/', params=params, headers=headers)

    if response.status_code == 200:
        return response.json()[post]
   
    retry = True
    while retry:
        if response.status_code != 200 and response.status_code != 429:
            retry = False
            print("some other ridiculus error idek dude...")
            print(response.json())
            return 'it broke'
        if response.status_code == 429:
            print("going to fast! 429 error!")
            wait_time = int(re.sub(r'[^\d+$]', '', response.json()['error']))
            print("sleeping for: " + str(wait_time) + " seconds...")
            time.sleep(wait_time + 2)
            response = requests.get(website + '_/api/chan/thread/', params=params, headers=headers)
            if response.status_code == 200:
                return response.json()[post]


def get_gallery(website=website, board='a', page='1'):
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'pleaseLet me Use this thing'
    }

    params = {
        'board': board,
        'page': page,
    }

    print("getting page:" + page)

    retry = True
    while retry:
        response = requests.get(
            website + '_/api/chan/gallery/',
            params=params,
            headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            threads = []
            for i in range(min(100, len(data))):
                op = data[i]['num']
                time_st = data[i]['timestamp']
                threads.append((op, time_st))
            return threads

        if response.status_code == 429:
            print("going too fast! 429 error!")
            wait_time = int(re.sub(r'[^\d+$]', '', response.json()['error']))
            print("sleeping for:", wait_time, "seconds...")
            time.sleep(wait_time + 2)
            continue

        # any other error
        retry = False
        print("some other ridiculous error idek dude...")
        print(response.json())
        return []
    return []


def write_posts(thread, board):
    def fillpostwrite(post):
        post_write = {
            'time': post["timestamp"],
            'board': post["board"]["shortname"],
            'threadID':post["thread_num"],
            'postID': post["num"],
            'text': post["comment"]
        }
        return json.dumps(post_write, indent=4) + '\n'
    #board = thread["op"]["board"]["shortname"]
    f = open(board + ".json", "a+")
    
    try:
        f.write(fillpostwrite(thread["op"]))
        print("writing thread num: " + thread["op"]["num"] + ", board: " + thread["op"]["board"]["shortname"] + " website: " + website)
        for post in thread["posts"]:
            f.write(fillpostwrite(thread["posts"][post]))
    except:
        print("some error writing...")
        print(json.dumps(thread, indent=5))




#approximatley track what page we are curerntly on 
#but not that usefull becasue boards gonna be moving about
page_track = {}
#set a default large time to get posts before-of
time_track = {}

for board in selected:
    page_track[board] = 1
    time_track[board] = 1768512543


#how many pages to do of a selected board at a time
pages_at_once = 30

#loop for all eternity
while 1:
    for board in selected:
        #loop from last page to next page
        for page in range(page_track[board], page_track[board] + pages_at_once):
            threads = get_gallery(board=board, page=str(page))
            time.sleep(60)
            latest_time = time_track[board]
            for thread in threads:
                if time_track[board] <= int(thread[0]):
                    page_track[board] += 1
                    continue
                if time_track[board] > thread[1]:
                    time_track[board] = thread[1]
            for thread in threads:
                write_posts(get_thread(post=thread[0], board=board), board=board)
                time.sleep(100)
        page_track[board] += pages_at_once

