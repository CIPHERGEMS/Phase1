# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 23:35:28 2020

@author: Dave
"""

from urllib.request import urlopen
from bs4 import BeautifulSoup
from time import sleep
import csv

from pathlib import Path
import os
import requests


home = os.path.expanduser('~')
homebase_path = Path(home)
os.chdir(homebase_path / "Desktop" / "scraping" )
save_path = homebase_path / "Desktop" / "Song_list_net" 
os.makedirs(save_path, exist_ok = True)

# url to scrape the songs list from
base_url = "http://www.song-list.net/{}/songs"

# artists list whose songs list is to be made
artists = ["21Savage"]

songs_dict = { }   

for artist in artists:
    
    artist_url = base_url.format(artist)
    print("Going to url : ", artist_url)
    
    html_page = requests.get(artist_url)
    soup = BeautifulSoup(html_page.content, 'html.parser')
    
    songs_list = soup.find('table', attrs={'class':'songs'}).find_all('td', attrs={'id':'songname'})
    
    songs_dict[artist] = []
    
    for song in songs_list:
        song_name = song.text.strip()
        songs_dict[artist].append(song_name)
        
    print("Artist : ", artist)
    print(songs_dict[artist][:5])
        
    sleep(10)


for key,val in songs_dict.items():
    print(key,len(val))
    
import json
json_file = str(save_path) + "/Artists-Songs Mapping.json"
with open(json_file, 'w') as file:
    json.dump(songs_dict, file)

with open(json_file) as f:
    a = json.load(f)
    print(a)
    
