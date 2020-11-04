# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:19:21 2020

@author: Dave
"""

# import libraries
import pandas as pd
import os
import xlrd
from pathlib import Path
from WordClouds import WordClouds
from NlpPreprocessing import NlpPreprocessing
import requests
import numpy as np
from PIL import Image

# grab input folders
home = os.path.expanduser('~')
homebase_path = Path(home)
country_folder = homebase_path / "Desktop" / "USA"
artists_tracks_folder = country_folder / "Got_Tracks"

# get mask used for wordcloud
mask = np.array(Image.open(requests.get("http://i.ebayimg.com/images/i/291262531593-0-1/s-l1000.jpg", stream=True).raw))
#mask = np.array(Image.open(requests.get("https://static.vecteezy.com/system/resources/previews/000/027/496/non_2x/us-map-silhouette-vector.jpg", stream=True).raw))

# initialize parameters for wordcloud
font = './Ignazio.ttf'
background = 'white'
color_generator = True
color = None
show = False
save = True

workbook_locations = xlrd.open_workbook(os.path.join(country_folder, 'USA Artists Locations.xlsx'))
sheet = workbook_locations.sheet_by_index(0)

years = None
years = [float(x) for x in list(range(1980, 2021))] 
search_year_type = "Single" # "Single" "Group" "None" wordcloud per yr, per 80's, or just all (None)
states = ['California','New York','Georgia','New Jersey','Virginia','Texas',
          'Pennsylvania','Ohio','Michigan','Louisiana','Illinois','Florida']

# Loop though the states to create the wordclouds
for state in states:
    
    #   Setup folders and initialize variables
    output_folder = homebase_path / "Desktop" / state
    os.makedirs(output_folder, exist_ok=True)
    full_clean_text = ''
    df_state_tracks = None
    
    # Get all tracks from state into one dataframe
    for rowx in range(sheet.nrows):
        if sheet.row_values(rowx)[2] == state:   
            df_artists = pd.read_csv(os.path.join(artists_tracks_folder, sheet.row_values(rowx)[0] + '.csv'),encoding='latin1')
            df_artists.columns = ['title','album','year','lyrics','urls']
            if df_state_tracks is not None:
                df_state_tracks = pd.concat([df_state_tracks, df_artists])
            else:
                df_state_tracks = df_artists
    
    #   Call WordCloud Class and create and save selected subset data
    # Subset the State Tracks by group of years.. e.g. get all the 90s tracks from New York        
    if search_year_type == "Group":
        str_yr_imgname = str(int(min(years))) + '_' + str(int(max(years))) + '.png'
        for ir in df_state_tracks.itertuples():
            if ir[3] != "UnKnown" and int(float(ir[3])) in years:
                NlpPrep = NlpPreprocessing(ir[4])
                clean_track = NlpPrep.start_preprocess()
                full_clean_text += clean_track + ' ' 
        outpath = output_folder / str_yr_imgname
        wc = WordClouds(full_clean_text, font, mask, background, color, color_generator=color_generator, show=show, save=save, savepath=outpath)
        wc.masking_wordcloud()
        wc.show_wordcloud()

    # Subset the State Tracks by year.. e.g. get all the tracks from New York in the year 1991 
    elif search_year_type == "Single":  
        for year in years:
            str_yr_imgname = str(int(year)) + '.png'
            for ir in df_state_tracks.itertuples():                 
                if ir[3] != "UnKnown" and int(float(ir[3])) == year:
                    NlpPrep = NlpPreprocessing(ir[4])
                    clean_track = NlpPrep.start_preprocess()
                    full_clean_text += clean_track + ' '  
            outpath = output_folder / str_yr_imgname
            wc = WordClouds(full_clean_text, font, mask, background, color, color_generator=color_generator, show=show, save=save, savepath=outpath)
            wc.masking_wordcloud()
            wc.show_wordcloud()
            
    # Subset the State Tracks by State only.. e.g. get all the tracks from New York out of the data we have            
    else:
        str_yr_imgname = 'NoYear' + '.png'
        for ir in df_state_tracks.itertuples():
            clean_trac = ir[4]
            NlpPrep = NlpPreprocessing(ir[4])
            clean_track = NlpPrep.start_preprocess()
            full_clean_text += clean_track + ' '
        outpath = output_folder / str_yr_imgname
        wc = WordClouds(full_clean_text, font, mask, background, color, color_generator=color_generator, show=show, save=save, savepath=outpath)
        wc.masking_wordcloud()
        wc.show_wordcloud()
    
    
