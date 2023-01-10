import pandas as pd
from user.models import Userinfo,Gameinfo

games = pd.read_csv('../static/File/game_preprocessed.csv')
# column_headers = list(games.columns.values)
# print(column_headers[5:])

print(len(games))

for i in range(0,len(games)):
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

    Gameinfo.objects.create(name=name, price=price,date=date,rate=rate,Singleplayer=Singleplayer,Action=Action,Multiplayer=Multiplayer,
                            Adventure=Adventure,Strategy=Strategy,Simulation=Simulation,Indie=Indie,RPG=RPG,Atmospheric=Atmospheric,Story_Rich=Story_Rich,
                            Open_World=Open_World,Casual=Casual,two_D=two_D,Sandbox=Sandbox,Fantasy=Fantasy,Online_Co_Op=Online_Co_Op,Exploration=Exploration,
                            three_D=three_D,Funny=Funny,Survival=Survival,Shooter=Shooter,Realistic=Realistic,Anime=Anime,Sci_fi=Sci_fi,FPS=FPS)

    # print(name)







