# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:47:18 2020

@author: Dave
"""

import os
from pathlib import Path

home = os.path.expanduser('~')
homebase_path = Path(home)
os.chdir(homebase_path / "Desktop")

#   text file of a list of artist name(cleaned no spaces, put '_' for ' ', '-' etc),artist id
in_file = open(os.path.join(main_path,'scripts_genuis','get_artist_id','artists_not_unique.txt'), "r")
#   text file to save same list above with id,name   no duplicates
outF = open(os.path.join(main_path,'scripts_genuis','get_artist_id','artists_unique.txt'), "w")
#   text file to save same list above with id,name   no duplicates (completely new artisits to scrape)
outF_new = open(os.path.join(main_path,'scripts_genuis','get_artist_id','artists_to_scrape.txt'), "w")

temp_dic = {}
for artist in in_file:
    artist = artist.rstrip('\n')
    if not artist:
        continue    
    artist = artist.strip()
    artist_id,name = artist.split(',')
    if artist_id not in temp_dic:
        temp_dic[artist_id] = name
    
for k, v in temp_dic.items():
    outF.write(str(k) + ','+ str(v) + '\n')
outF.close()
in_file.close()

#%%
import pickle
#   update artitst id dictionary
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#   dict of unique artists already scraped
if os.path.exists(os.path.join(main_path,'scripts_genuis','get_artist_id','artist_id_dic.pkl')):
    org_dic = load_obj(os.path.join(main_path,'scripts_genuis','get_artist_id','artist_id_dic')) 
else:
    print('-----------No Dictionary-------------')
    org_dic = {}
    
#%%
for k, v in temp_dic.items():
    if k not in org_dic:
        outF_new.write(str(k) + ','+ str(v) + '\n')
        org_dic[k] = v
outF_new.close()

#%%
save_obj(org_dic,os.path.join(main_path,'scripts_genuis','get_artist_id','artist_id_dic'))         