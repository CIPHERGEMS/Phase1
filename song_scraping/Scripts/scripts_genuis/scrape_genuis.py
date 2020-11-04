# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 14:16:44 2020

@author: Dave
"""

import pandas as pd
import time
import lyricsgenius
client_access_token = "J3_2xfJqz3MKGav2vPVvTCKCtsOJhE9jmafJbBR3VOqwmYuuB1z9613WlT8JqbcZ"

import os
import xlrd
from pathlib import Path

home = os.path.expanduser('~')
homebase_path = Path(home)
scraping_folder = homebase_path / "Desktop" / "song_scraping"
os.chdir(scraping_folder)

genius = lyricsgenius.Genius(client_access_token, remove_section_headers=True, sleep_time=1.5, timeout=10,
                 skip_non_songs=True, excluded_terms=["Remix", "Live", "Edit", "Mix", "Club"])

for dirname, _, filenames in os.walk(scraping_folder / "Artist_to_scrape"):
    for filename in filenames:
        workbook = xlrd.open_workbook(os.path.join(dirname, filename))
        sheet = workbook.sheet_by_index(0)
        for rowx in range(sheet.nrows):
            current_artist = sheet.row_values(rowx)[0]
            print(current_artist)
            #Empty lists for artist, title, album and lyrics information
            titles = []
            albums = []
            years = []
            lyrics = []
            urls = []
            #Search for max_songs = n and sort them by popularity
            artist = genius.search_artist(current_artist, include_features=False)
            songs = artist.songs
            if len(songs) == 0:
                outF = open(os.path.join(scraping_folder, "all_tracks_from_sites_with_duplicates", filename.split('_')[0], current_artist+'__genius.txt'), "w")
                outF.write('')
                outF.close()
            else:           
                #Append all information for each song in the previously created lists
                for song in songs:
                    if song is not None:
                        titles.append(song.title)
                        if song.album is not None:
                            albums.append(song.album)
                        else:
                            albums.append(None)
                        if song.year is not None:
                            years.append(song.year[0:4])
                        else:
                            years.append(None)
                        lyrics.append(song.lyrics)
                        urls.append(song._url)
                
                #Create a dataframe for our collected tracklist   
                tracklist = pd.DataFrame({'title':titles, 'album':albums, 'year':years, 'lyrics':lyrics, 'urls':urls})   
                os.makedirs(os.path.join(scraping_folder, "all_tracks_from_sites_with_duplicates", filename.split('_')[0]),exist_ok=True)
                #Save the final tracklist to csv format
                tracklist.to_csv(os.path.join(scraping_folder, "all_tracks_from_sites_with_duplicates", filename.split('_')[0], current_artist+'__genius.csv'), encoding = 'utf-8', index=False)

#%%
