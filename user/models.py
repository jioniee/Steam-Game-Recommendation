from django.db import models


# Create your models here.
class Userinfo(models.Model):
    name = models.CharField(max_length=32)
    # password = models.CharField(max_length=64)
    link = models.CharField(max_length=200)

class Gameinfo(models.Model):
    name = models.CharField(max_length=100)
    price = models.FloatField()
    date = models.IntegerField()
    rate = models.FloatField()
    Singleplayer = models.IntegerField(default=0)
    Action = models.IntegerField(default=0)
    Multiplayer = models.IntegerField(default=0)
    Adventure = models.IntegerField(default=0)
    Strategy = models.IntegerField(default=0)
    Simulation = models.IntegerField(default=0)
    Indie = models.IntegerField(default=0)
    RPG = models.IntegerField(default=0)
    Atmospheric = models.IntegerField(default=0)
    Story_Rich = models.IntegerField(default=0)
    Open_World = models.IntegerField(default=0)
    Casual = models.IntegerField(default=0)
    two_D = models.IntegerField(default=0)
    Sandbox = models.IntegerField(default=0)
    Fantasy = models.IntegerField(default=0)
    Online_Co_Op = models.IntegerField(default=0)
    Exploration = models.IntegerField(default=0)
    three_D = models.IntegerField(default=0)
    Funny = models.IntegerField(default=0)
    Survival = models.IntegerField(default=0)
    Shooter = models.IntegerField(default=0)
    Realistic = models.IntegerField(default=0)
    Anime = models.IntegerField(default=0)
    Sci_fi = models.IntegerField(default=0)
    FPS = models.IntegerField(default=0)

class Usergame(models.Model):
    username = models.CharField(max_length=32)
    gamename = models.CharField(max_length=100)
    time = models.FloatField()

