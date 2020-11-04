# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 21:37:32 2020

@author: Dave
"""

import os
import xlrd
from pathlib import Path
import pandas as pd
import numpy as np

home = os.path.expanduser('~')
homebase_path = Path(home)
data_folder = homebase_path / "Desktop" / "song_scraping" / "all_tracks_from_sites_with_duplicates" / "USA2"
out_folder = homebase_path / "Desktop" / "song_scraping" / "all_tracks_from_sites_with_duplicates" / "USA2"

def mode(x):
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m]

for dirname, _, filenames in os.walk(data_folder):
    for filename in filenames:
        df = pd.read_csv(os.path.join(dirname, filename), encoding='cp1252')

#%%
        #    Get dict of knwon years
        df = df.where(pd.notnull(df), 'UnKnown')
        temp_dic = {}
        for i in range(len(df)): 
            if df.loc[i, "album"] != 'UnKnown' and df.loc[i, "year"] != 'UnKnown':
                if df.loc[i, "album"] not in temp_dic:            
                    temp_dic[df.loc[i, "album"]] = [df.loc[i, "year"]]
                else:
                    temp_dic[df.loc[i, "album"]] += [df.loc[i, "year"]]
            
#%%
        #   replace known years
        for i in range(len(df)): 
            if df.loc[i, "year"] == 'UnKnown' and df.loc[i, "album"] in temp_dic:
                new_year = int(mode(np.array(temp_dic[df.loc[i, "album"]])))
                df.loc[i, "year"] = new_year
                       
#%%
        #   Remove tracks with less than 50 words
        rows_keep = []
        for i in range(len(df)): 
            if len(df.loc[i, "lyrics"].strip().split()) >= 50:
                rows_keep.append(i)
        
        df1 = df.loc[rows_keep, :]
        
#%%
        #   Remove tracks with annoted in url
        rows_keep = []
        for i in range(len(df1)): 
            if "annotated" not in df1.loc[i, "urls"]:
                rows_keep.append(i)
        
        df = df1.loc[rows_keep, :]
        df.to_csv(os.path.join(out_folder,filename.split('__')[0]),encoding='utf-8-sig',  index=False)
        
