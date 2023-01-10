# Example link, user
testlink = "https://steamcommunity.com/profiles/76561198322599801/games/?tab=all"
testuser = "jionie"

import json
import urllib.request
from lxml import etree
import re
from bs4 import BeautifulSoup
from user.models import Userinfo,Gameinfo,Usergame

def create_request(link):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36'
    }

    url = link

    request = urllib.request.Request(url = url, headers=headers)
    return request

def get_content(request):
    response = urllib.request.urlopen(request)
    content = response.read().decode('utf-8')
    return content

def save_games(content):
    print(content)

    for i in range(len(content)):
        print(content[i:i+7])
        if content[i:i+7] == "rgGames":
            break

    rgGames = ""
    index = i+10
    for i in range(index,len(content)):
        if content[i] != ';':
            rgGames = rgGames + content[i]
        else:
            break

    data_json = json.loads(rgGames, strict=False)

    user = testuser
    # if user already exists:
    userExist = False
    alluser = Userinfo.objects.all()
    for obj_user in alluser:
        if obj_user.name == user:
            userExist = True
            break



# if __name__ == '__main__':


def loadUsergame(link):
    link = testlink

    request = create_request(link)

    content = get_content(request)

    save_games(content)
