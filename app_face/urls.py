from django.urls import path
from app_face.views import yuznianiqlash

urlpatterns = [
    path('video_feed/', yuznianiqlash.video_feed, name='video_feed'),
    path('face/', yuznianiqlash.face, name='face'),
    path('', yuznianiqlash.home, name='home'),
    path('main/', yuznianiqlash.main, name='main'),

]
