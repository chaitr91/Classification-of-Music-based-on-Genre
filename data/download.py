import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import urllib.request
import os
import re
from slugify import slugify

SPOTIPY_CLIENT_ID="8aea0035daa540e6ae9dba92667fb68b"
SPOTIPY_CLIENT_SECRET="2d8a6fd9ebff4837a677efe0b00256af"

client_credentials_manager = SpotifyClientCredentials(SPOTIPY_CLIENT_ID,SPOTIPY_CLIENT_SECRET)
spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
genreList = spotify.recommendation_genre_seeds()
for i in genreList['genres']:
	print(i)
	if not os.path.exists('./'+i):
	    os.makedirs('./'+i)
	musicList = spotify.recommendations(seed_genres=[i],limit=30,country='US')
	for j in musicList['tracks']:
		#print(j['album']['name'])
		#print(j['preview_url'])
		if(j['preview_url']):
			filename = slugify(str(j['album']['name']))
			fullfilename = os.path.join('./'+ i + '/', filename + '.mp3')
			urllib.request.urlretrieve (j['preview_url'], fullfilename)
