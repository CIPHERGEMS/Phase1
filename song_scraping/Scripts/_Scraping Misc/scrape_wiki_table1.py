# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 17:48:36 2020

@author: Dave
"""


import os
from pathlib import Path
from googlesearch import search 
import requests
import time
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from urllib.request import urlopen
import pickle

#   update artitst id dictionary
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def get_dicts(file_path):
    if os.path.exists(file_path):
        return load_obj(file_path) 
    else:
        return {}

home = os.path.expanduser('~')
homebase_path = Path(home)
outfolder = homebase_path /'Desktop' / 'CIPHER' / 'scripts_genuis' / 'artist_textfiles'
os.chdir(outfolder)  

output = outfolder / 'artist_dictionaries'
os.makedirs(output, exist_ok=True)
artist_dict = get_dicts(outfolder / 'artist_dictionaries' / 'artist_dic.pkl')
albums_dict = get_dicts(outfolder / 'artist_dictionaries' / 'album_dic.pkl')
years_dict = get_dicts(outfolder / 'artist_dictionaries' / 'years_dic.pkl')
discography_dict = get_dicts(outfolder / 'artist_dictionaries' / 'discography_dic.pkl')

no_album = outfolder / "no_album_artists.txt"
outFile_no_album = open(no_album, "a")

no_link = outfolder / "no_link_artists.txt"
outFile_no_link = open(no_link, "a")

file_list = 'test' # rollingstone , puddingcool
file = str(outfolder) + '/' + file_list + ".txt"
list_of_artists = []
f = open(file, "r")
for artist in f:
    artist_row = artist.split(',')
    if len(artist_row) == 1:
        artist = artist_row[0].rstrip('\n')
    else:
        artist = artist_row[1].rstrip('\n')
        artist = artist.replace('_',' ')
    if not artist:
        continue    
    list_of_artists.append(artist)
f.close()

#%%
for artist in list_of_artists:
  #  print(artist)
    if len(discography_dict) > 0:
        Keymax = max(albums_dict)
        new_id = Keymax + 1
    else:
        new_id = 0

    query = 'wiki ' + artist + ' discography'
    url_list = [j for j in search(query, tld="com", num=5, stop=1, pause=2)]
    for url in url_list:
        if 'discography' in url:
            break
#%%
page = urlopen(url)
soup = BeautifulSoup(page, "html.parser")
tables = soup.find_all("table", class_='wikitable')

tables = soup.find_all("h2")

#%%
for table in tables:
    for th in table.find_all('span'):
        print(th)
        #%%
    for tr in table.find_all('tr')[0:]:
        tds = tr.find_all('td')
        print(tds)
        
    break

#%%
    
    article = urlopen(url)
    soup = BeautifulSoup(article, 'lxml')
    tables = soup.find_all('table')
    if len(tables) < 1:
        outFile_no_album.write(artist+"\n")
    
    # Search through the tables for the one with the headings we want.
    for table in tables:
        print(str(table),'\n\n')
        break
        if 'mixtape' in  str(table).split('>')[2]:
            pass
        elif 'EPs' in  str(table).split('>')[2]:
            pass
        elif 'album' in str(table).split('>')[2]:
            if 'single' in str(table).split('>')[2]:
                continue
            if 'song' in str(table).split('>')[2]:
                continue
        else:
            continue
    
        rows = table.find_all('tr')
        for i in range(2,len(rows)):
            a = str(rows[i].find('th')) 
            if a == 'None':
                continue
            else:
                tds = rows[i].find_all('td')
                for td in tds[:1]:
                    td = str(td).split('Released:')
                    if len(td) == 1:
                        released_date = 'TBA_or_unknown'
                    else:
                        released_date = td[1].split('<')[0].strip()

                if len(a.split('<i>')) == 1:
                    break
                namelist = a.split('<i>')[1].split('</i>')[0].split('</a>')
                if len(namelist) == 1: 
                    link = ''
                    title = namelist[0].strip()
                elif len(namelist) == 2:
                    name = namelist[0].replace('"mw-redirect"','')
                    _, link, _, title, _ = name.split('"')
                    link = str("https://en.wikipedia.org" + link).strip()
                    title = title.strip()
                else:
                    exit('namelist is wrong')
             
            if link == '':
                outFile_no_link.write(artist+','+title+','+released_date+"\n")
            else:
                albums_dict[new_id] = title
                artist_dict[new_id] = artist
                years_dict[new_id] = released_date
                discography_dict[new_id] = link
                new_id += 1
            
outFile_no_album.close()    
outFile_no_link.close()    