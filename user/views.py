import openpyxl as openpyxl
import pandas as pd
from django.shortcuts import render, HttpResponse, redirect
import os
import random

# Create your views here.
from sklearn.preprocessing import MinMaxScaler

from user.models import Userinfo, Gameinfo, Usergame
from user.machineRec import prediction, get_u_vector, pca_trans, hopkins_statistic, get_k, get_minmax, get_gameno, time_scaler, time_dropper, get_u_data, collaborative_filtering


def index(request):
    return HttpResponse("helloworld")


# User signup
def welcome(request):
    if request.method == "GET":
        return render(request, 'welcome.html')

    # Retrieve user's submission
    user = request.POST.get("user")
    # password = request.POST.get("password")
    link = request.POST.get("link")

    isEmpty = False
    # print(user , " user is ")
    if user == "" or link == "":
        message = "Please fill the blank"
        isEmpty = True
        return render(request, 'welcome.html', locals())

    getrequest = create_request(link)

    content = get_content(getrequest)

# -------------------
    userExist = False
    linkExist = False
    alluser = Userinfo.objects.all()

    for obj_user in alluser:
        if obj_user.name == user:
            message = "user already exist"
            userExist = True
            break
        if obj_user.link == link:
            message = "link is already used"
            linkExist = True
            break

    # If user already exist:
    # Add a new page to tell if user want to refresh the account
    if userExist or linkExist:
        # Replace with a page
        # print("exists")
        return render(request, 'welcome.html', locals())

    data_json = save_games(content, user, userExist, request)
    # print(data_json)
    u_games=[]
    u_time = []
    u_link = []
    u_image = []
    for obj in data_json:
        # print("this is obj")
        # print(obj)
        u_games.append(obj['name'])
        if 'hours_forever' in obj:
            u_time.append(obj['hours_forever'])
        else:
            u_time.append(0.0)
        # u_link.append(obj['name'])
        u_image.append(obj['logo'])

    randnumlist = random.sample(range(1,34),5)
    randnumber1 = randnumlist[0]
    randnumber2 = randnumlist[1]
    randnumber3 = randnumlist[2]
    randnumber4 = randnumlist[3]
    randnumber5 = randnumlist[4]
    img1 = '../static/image/image' + str(randnumber1) + '.jpg'
    # print(img1)
    img2 = '../static/image/image' + str(randnumber2) + '.jpg'
    img3 = '../static/image/image' + str(randnumber3) + '.jpg'
    img4 = '../static/image/image' + str(randnumber4) + '.jpg'
    img5 = '../static/image/image' + str(randnumber5) + '.jpg'
    # Add to database
    Userinfo.objects.create(name=user, link=link)

    if not u_games:
        return render(request, 'gotosteam.html')

    allgame = []
    gameinfo_all = Gameinfo.objects.all()
    for obj in gameinfo_all:
        allgame.append(obj.name)


    for i in range(len(data_json)):
        if 'hours_forever' in data_json[i]:
            time_raw = data_json[i]['hours_forever']
            print(type(time_raw))
            time = time_raw.replace(",", "")
        else:
            time = 0

        if data_json[i]['name'] in allgame:
            Usergame.objects.create(username=user, gamename=data_json[i]['name'], time = time)
        # Replace with a page
    # Transfer to a new page to show recommendation data

    # userGameTimeAll = Usergame.objects.all()
    # usergame_alluser = []
    # all_users_thisUserTime = []
    #
    # for obj in alluser:
    #     usergame_alluser.append(obj.name)
    #
    # for user in usergame_alluser:
    #     usergame_thisusertime = [0] * len(allgame)
    #     for game in allgame:
    #         for obj in userGameTimeAll:
    #             if obj.gamename == game
    #
    # for obj in userGameTimeAll:
    #     for game in allgame:
    #         if obj.gamename == game:
    #             usergame_thisusertime.append(obj.time)
    #         else:
    #             usergame_thisusertime.append(0)
    df = pd.read_csv('./static/File/game_preprocessed.csv', index_col=[0])
    game_names = df['Name'].tolist()

    data = df.drop(['Name', 'Rate'], axis=1)
    features = data.columns.tolist()
    data = MinMaxScaler().fit_transform(data)
    data = pd.DataFrame(data)



    u_times = getalluser()
    u_times = time_dropper(u_times,game_names)
    time_scaler(u_times)

    scores, pca = pca_trans(data, features)
    u_data = get_u_data(u_times,pca,df,data)

    u_vector = get_u_vector(u_games, pca, df, data)

    rec_games_2 = collaborative_filtering(u_data, u_times,u_vector,df)
    print("this is rec2: ")
    print(rec_games_2)


    # df = pd.read_csv('./static/File/game_preprocessed.csv', index_col=[0])
    # data = df.drop(['Name', 'Rate'], axis=1)
    # data = MinMaxScaler().fit_transform(data)
    # data = pd.DataFrame(data)

    # print(u_games)



    rec_games_1 = prediction([u_vector], scores)
    print("this is rec1: ")
    print(rec_games_1)

    rec_games = []
    for games in rec_games_1:
        if games not in rec_games:
            rec_games.append(games)

    if rec_games_2:
        for games in rec_games_2:
            if games not in rec_games:
                rec_games.append(games)

    name = []
    price = []
    date = []
    rate = []
    num = 0
    for game in rec_games:
        # print("this game is recommed")
        # print(game)
        gamename = df.loc[game,'Name']
        if gamename not in u_games:
            num = num + 1
            # print("this game is new")
            # print("num is: ", num)
            if num ==1:
                name1 = df.loc[game,'Name']
                price1 = df.loc[game, 'Price']
                price1 = str(price1)
                price1 = "Price : " + price1
                date1 = df.loc[game, 'Date']
                rate1 = df.loc[game, 'Rate']
                rate1 = str(rate1)
                rate1 = "Rate : " + rate1
                link1 = f'https://store.steampowered.com/search/?term={name1}'
            if num ==2:
                name2 = df.loc[game,'Name']
                price2 = df.loc[game, 'Price']
                price2 = str(price2)
                price2 = "Price : " + price2
                date2 = df.loc[game, 'Date']
                rate2 = df.loc[game, 'Rate']
                rate2 = str(rate2)
                rate2 = "Rate : " + rate2
                link2 = f'https://store.steampowered.com/search/?term={name2}'
            if num ==3:
                name3 = df.loc[game,'Name']
                price3 = df.loc[game, 'Price']
                price3 = str(price3)
                price3 = "Price : " + price3
                date3 = df.loc[game, 'Date']
                rate3 = df.loc[game, 'Rate']
                rate3 = str(rate3)
                rate3 = "Rate : " + rate3
                link3 = f'https://store.steampowered.com/search/?term={name3}'
            if num ==4:
                name4 = df.loc[game,'Name']
                price4 = df.loc[game, 'Price']
                price4 = str(price4)
                price4 = "Price : " + price4
                date4 = df.loc[game, 'Date']
                rate4 = df.loc[game, 'Rate']
                rate4 = str(rate4)
                rate4 = "Rate : " + rate4
                link4 = f'https://store.steampowered.com/search/?term={name4}'
            if num ==5:
                name5 = df.loc[game,'Name']
                price5 = df.loc[game, 'Price']
                price5 = str(price5)
                price5 = "Price : " + price5
                date5 = df.loc[game, 'Date']
                rate5 = df.loc[game, 'Rate']
                rate5 = str(rate5)
                rate5 = "Rate : " + rate5
                link5 = f'https://store.steampowered.com/search/?term={name5}'
            if num == 6:
                break
            # print(df.loc[game, ['Name', 'Price', 'Date', 'Rate']])


    return render(request, 'recommendation.html', locals())

    # return HttpResponse("User signup successfully")

# -------------------
    # return HttpResponse("sign up successfully !!!")


# show user
def userlist(request):
    user_list = Userinfo.objects.all()
    # print(user_list)

    return render(request, "userlist.html", {"user_list": user_list})

# store game data
def gamesdata(request):
    # print(os.getcwd())
    games = pd.read_csv('./static/File/game_preprocessed.csv')

    for i in range(0, len(games)):
        name = games['Name'][i]
        price = games['Price'][i]
        date = games['Date'][i]
        rate = games['Rate'][i]
        Singleplayer = games['Singleplayer'][i]
        Action = games['Action'][i]
        Multiplayer = games['Multiplayer'][i]
        Adventure = games['Adventure'][i]
        Strategy = games['Strategy'][i]
        Simulation = games['Simulation'][i]
        Indie = games['Indie'][i]
        RPG = games['RPG'][i]
        Atmospheric = games['Atmospheric'][i]
        Story_Rich = games['Story Rich'][i]
        Open_World = games['Open World'][i]
        Casual = games['Casual'][i]
        two_D = games['2D'][i]
        Sandbox = games['Sandbox'][i]
        Fantasy = games['Fantasy'][i]
        Online_Co_Op = games['Online Co-Op'][i]
        Exploration = games['Exploration'][i]
        three_D = games['3D'][i]
        Funny = games['Funny'][i]
        Survival = games['Survival'][i]
        Shooter = games['Shooter'][i]
        Realistic = games['Realistic'][i]
        Anime = games['Anime'][i]
        Sci_fi = games['Sci-fi'][i]
        FPS = games['FPS'][i]
        # print(rate)

        print(name)
        Gameinfo.objects.create(name=name, price=price, date=date, rate=rate, Singleplayer=Singleplayer, Action=Action,
                                Multiplayer=Multiplayer,
                                Adventure=Adventure, Strategy=Strategy, Simulation=Simulation, Indie=Indie, RPG=RPG,
                                Atmospheric=Atmospheric, Story_Rich=Story_Rich,
                                Open_World=Open_World, Casual=Casual, two_D=two_D, Sandbox=Sandbox, Fantasy=Fantasy,
                                Online_Co_Op=Online_Co_Op, Exploration=Exploration,
                                three_D=three_D, Funny=Funny, Survival=Survival, Shooter=Shooter, Realistic=Realistic,
                                Anime=Anime, Sci_fi=Sci_fi, FPS=FPS)

    return HttpResponse("add database successfully")

import json
import urllib.request
from lxml import etree
import re
from bs4 import BeautifulSoup
from user.models import Userinfo, Gameinfo, Usergame


def create_request(link):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36'
    }

    url = link

    request = urllib.request.Request(url=url, headers=headers)
    return request


def get_content(request):
    response = urllib.request.urlopen(request)
    content = response.read().decode('utf-8')
    return content


def save_games(content, user, userExist, request):
    print(content)

    for i in range(len(content)):
        # print(content[i:i + 7])
        if content[i:i + 7] == "rgGames":
            break

    rgGames = ""
    index = i + 10
    for i in range(index, len(content)):
        if content[i] != ';':
            rgGames = rgGames + content[i]
        else:
            break
    if rgGames == "":
        return render(request, 'gotosteam.html')

    # print("SSSSSSSSSSSSSSS")
    # print(rgGames)
    return json.loads(rgGames, strict=False)

    # user = testuser

def recommendation(request):
    return render(request, 'recommendation.html')

def clear(request):
    # Gameinfo.objects.all().delete()
    Userinfo.objects.all().delete()
    Usergame.objects.all().delete()
    return HttpResponse("delete successfully")

def gotosteam(request):
    return render(request, 'gotosteam.html')

def getalluser():
    users = Userinfo.objects.all()
    allUserName = []
    # print("This is all user: ")
    for obj in users:
        allUserName.append(obj.name)
        # print(obj.name)

    userGames = Usergame.objects.all()
    allUserGames = []
    for user in allUserName:
        thisUserGame = []
        for usergame in userGames:
            if usergame.username == user:
                thisUserGame.append(usergame.gamename)
        allUserGames.append(thisUserGame)

    allUserTimes = []
    for user in allUserName:
        thisUserTime = []
        for usergame in userGames:
            if usergame.username == user:
                thisUserTime.append(usergame.time)
        allUserTimes.append(thisUserTime)

    allgame = []
    gameinfo_all = Gameinfo.objects.all()
    for obj in gameinfo_all:
        allgame.append(obj.name)

    times = pd.DataFrame(columns=list(allgame), index=allUserName)
    for usergame_index in range(len(allUserGames)):
        i = 0
        for game in allUserGames[usergame_index]:
            times.loc[allUserName[usergame_index], game] = allUserTimes[usergame_index][i]
            i+=1
    print("hello this is times: ")
    print(times)


    # [1wan,2wan,3wan]
    # [[Ayou,Byou],[Byou,Cyou],[Ayou]]
    # [[1,2],[500,1],[4]]

    # print("User own games")
    # print(allUserGames[0])
    # print(allUserGames[1])
    # print(allUserTimes[0])
    # print(allUserTimes[1])
    # all_usergames1 = set(allUserGames[0])
    # all_usergames2 = set(allUserGames[1])
    # all_users = all_usergames1 | all_usergames2
    #
    # times = pd.DataFrame(columns=list(all_users), index=allUserName)
    # i = 0
    # for game in allUserGames[0]:
    #     times.loc[allUserName[0], game] = allUserTimes[0][i]
    #     # print(game, allUserTimes[0][i])
    #     i += 1
    # i = 0
    # for game in allUserGames[1]:
    #     times.loc[allUserName[1], game] = allUserTimes[1][i]
    #     i += 1
    # print(times.head(5))


    # print(allUserGames[2])
    return times


