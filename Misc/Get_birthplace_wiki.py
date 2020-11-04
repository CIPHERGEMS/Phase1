# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:22:27 2020

@author: Dave
"""

from urllib.request import urlopen
import wikipedia as w
from bs4 import BeautifulSoup as bs
import re
import os
import xlrd
from pathlib import Path
import pandas as pd
import numpy as np

w.set_lang('en')
home = os.path.expanduser('~')
homebase_path = Path(home)
data_folder = homebase_path / "Desktop" / "song_scraping" / "all_tracks_from_sites_with_duplicates" / "USA2"

for dirname, _, filenames in os.walk(data_folder):
    for filename in filenames:
        artist = filename.split('.')[0]
        print(artist)
        artist_wiki = w.page(w.search(artist)[0])
        page = bs(urlopen(artist_wiki.url).read(), 'lxml')
     
        POB = page.find_all('div', {'class': 'birthplace'})[0]
        print(POB.get_text())        
        
    