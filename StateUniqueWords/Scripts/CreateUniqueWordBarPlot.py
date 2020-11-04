# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 23:58:30 2020

@author: Dave
"""


# import libraries
import pandas as pd
import os
import xlrd
from pathlib import Path
import numpy as np
from PIL import Image
from NlpPreprocessing import NlpPreprocessing
import random
import matplotlib.pyplot as plt

# grab input folders
home = os.path.expanduser('~')
homebase_path = Path(home)
country_folder = homebase_path / "Desktop" / "USA"
artists_tracks_folder = country_folder / "Got_Tracks"

workbook_locations = xlrd.open_workbook(os.path.join(country_folder, 'USA Artists Locations.xlsx'))
sheet = workbook_locations.sheet_by_index(0)

years = [x for x in list(range(1980, 2021))] 
search_year_type = "Single" # "Single" "Group" "None" wordcloud per yr, per 80's, or just all (None)
states = ['California','New York','Georgia','New Jersey','Virginia','Texas',
          'Pennsylvania','Ohio','Michigan','Louisiana','Illinois','Florida']

# Loop though the states and save in dict
dict_state_tracks = {}
min_number_tracks = 1e16
ThresholdNumberTracks = 10

if search_year_type == "Single":
    yearsloop = years
else:
    yearsloop = [0]
for year in yearsloop:
    for state in states:
        
        #   Setup folders and initialize variables
        df_state_tracks = None
        list_tracks = []
        
        # Get all tracks from state into one dataframe
        for rowx in range(sheet.nrows):
            if sheet.row_values(rowx)[2] == state:   
                df_artists = pd.read_csv(os.path.join(artists_tracks_folder, sheet.row_values(rowx)[0] + '.csv'),encoding='latin1')
                df_artists.columns = ['title','album','year','lyrics','urls']
                if df_state_tracks is not None:
                    df_state_tracks = pd.concat([df_state_tracks, df_artists])
                else:
                    df_state_tracks = df_artists            
     
        #   Create and save selected subset data
        # Subset the State Tracks by group of years.. e.g. get all the 90s tracks from New York        
        if search_year_type == "Group":
            str_yr_imgname = str(int(min(years))) + '_' + str(int(max(years))) + '.png'
            for ir in df_state_tracks.itertuples():
                if ir[3] != "UnKnown" and int(float(ir[3])) in years:
                    NlpPrep = NlpPreprocessing(ir[4])
                    clean_track = NlpPrep.start_preprocess()
                    list_tracks.append(clean_track) 
     
        # Subset the State Tracks by year.. e.g. get all the tracks from New York in the year 1991 
        elif search_year_type == "Single":  
            str_yr_imgname = str(int(year)) + '.png'
            for ir in df_state_tracks.itertuples():                 
                if ir[3] != "UnKnown" and int(float(ir[3])) == year:
                    NlpPrep = NlpPreprocessing(ir[4])
                    clean_track = NlpPrep.start_preprocess()
                    list_tracks.append(clean_track) 
    
        # Subset the State Tracks by State only.. e.g. get all the tracks from New York out of the data we have            
        else:
            str_yr_imgname = 'NoYear' + '.png'
            for ir in df_state_tracks.itertuples():
                clean_trac = ir[4]
                NlpPrep = NlpPreprocessing(ir[4])
                clean_track = NlpPrep.start_preprocess()
                list_tracks.append(clean_track) 
          
        # Update minimum track number and store track lists into dict
        if len(list_tracks) > ThresholdNumberTracks and len(list_tracks) < min_number_tracks:
            min_number_tracks = len(list_tracks)
        if len(list_tracks) != 0:
            dict_state_tracks[state] = list_tracks
    
    if len(dict_state_tracks) > 0:
        dict_unique_words = {}
        for state, list_tracks in dict_state_tracks.items():
            track_str = ''
            random.shuffle(list_tracks)
            tracks = list_tracks[:min_number_tracks]
            for track in tracks:
                track_str += track + ' '
            dict_words = {}
            for word in track_str.split():
                if word not in dict_words:
                    dict_words[word] = 1
                else:
                    dict_words[word] += 1
            dict_unique_words[state] = len(dict_words)/min_number_tracks
          
        keys = dict_unique_words.keys()
        values = dict_unique_words.values()
        
        # Plotting
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        plt.bar(keys, values)
        ax.set_xticklabels(keys, rotation = 90)
        plt.xlabel('States')
        plt.ylabel('Average number of unique words per track')
        plt.title(str(year))
        fig.savefig(homebase_path / "Desktop" / str_yr_imgname, dpi=300, format='png', bbox_inches='tight')
        fig.clear()
        plt.close(fig)