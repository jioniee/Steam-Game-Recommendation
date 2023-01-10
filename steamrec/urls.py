"""steamrec URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.template.defaulttags import url
from django.urls import path, re_path

import steamrec.urls
from user import views

urlpatterns = [
    # Just for testing
    re_path(r'^$', views.welcome),
    path('index/', views.index),

    # welcome -- first page. User input their user & link -> recommendation
    path('welcome/', views.welcome),

    # show user, for testing
    path('userlist/', views.userlist),

    # add games to database, for developer
    path('gamesdata/', views.gamesdata),

    # show recommendation result.
    path('recommendation/', views.recommendation),

    # Clear data in the database, for developer
    path('clear/', views.clear),

    # If user have no game, send them to Steam
    path('gotosteam/', views.gotosteam),

    # Show User own games, For developers
    # path('getalluser/', views.getalluser),
]
