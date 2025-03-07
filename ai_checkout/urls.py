from django.contrib import admin
from django.urls import path, include
from ai_checkout import views

urlpatterns = [
    path('', views.landingpage, name='landingpage'),
    path('main/', views.main, name='main'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('result_table/', views.result_table, name='result_table'),
    

]