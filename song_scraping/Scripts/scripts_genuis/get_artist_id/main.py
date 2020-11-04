# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:22:25 2020

@author: Dave
"""

exp = 'get_artist_id'
in_file = "artists_not_unique.txt"

import lyricsgenius as genius
import urllib.request
import urllib.parse
import urllib.error
from bs4 import BeautifulSoup
import ssl
import json
import ast
import os
import re 
import sys
import copy as cp
import pickle
from urllib.request import Request, urlopen
from pathlib import Path

home = os.path.expanduser('~')
homebase_path = Path(home)

os.chdir(homebase_path /'Desktop' / 'CIPHER' / 'scripts_genuis' / exp)  

api = genius.Genius('J3_2xfJqz3MKGav2vPVvTCKCtsOJhE9jmafJbBR3VOqwmYuuB1z9613WlT8JqbcZ')
outF = open(homebase_path /'Desktop' /  'CIPHER' / 'scripts_genuis' / exp / "artists_not_unique1.txt", "a")
f = open(in_file, "r")
for artist in f:
    artist = artist.rstrip('\n')
    if not artist:
        continue    
    artist = artist.strip()
    artist_obj = api.search_artist(artist, max_songs=1)
    outF.write(str(artist_obj._id)+','+artist+"\n")
outF.close()
    
