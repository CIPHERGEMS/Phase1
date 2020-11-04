# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 22:12:19 2020

@author: Dave
"""

import os
import pickle
import re
import pandas as pd
from uuid import getnode as get_mac
 
def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def keep_alphanumeric_space_underscore(string):
    string = re.sub(r'[^A-Za-z0-9 ]+', '', string)
    string = string.replace (" ", "_")
    string = string.replace ("__", "_")
    return string

def keep_alphanumeric_space(string):
    return re.sub(r'[^A-Za-z0-9 ]+', '', string)

def main():  
    mac = get_mac()
    main_path = "/home/dbrowne/"
    if mac == 45015790230696:  # dave's laptop
        main_path = "C:/Users/dbrowne/"
    if mac == 198112232959511:  # dave's home
        main_path = "C:/Users/dave/"
    
    data_folder = 'Genius'
    artist_path_lst = [f.path for f in os.scandir(os.path.join(main_path,'Desktop','music_data',data_folder)) if f.is_dir() ] 
    
    year_dict = {}
    artist_dict = {}
    title_dict = {}
    album_dict = {}
    for artist_path in artist_path_lst:
        _, artist_name = os.path.split(artist_path)
        #   text file to save artist tracks- id,name,album
        out_textfiles = os.path.join(artist_path,'text_song_files')
        os.makedirs(out_textfiles,exist_ok=True)
        outF = open(os.path.join(artist_path,artist_name+'_tracklist.txt'), "w")
        metadata_dict = load_obj(artist_path + '/' +'metadata.pkl')
        for song_id,song_meta_dict in metadata_dict.items():
            lyrics = song_meta_dict['lyrics']        
            year = (song_meta_dict['release_date'].split('-'))[0]
            album = song_meta_dict['album']
            album = keep_alphanumeric_space_underscore(album)
            title = song_meta_dict['title']
            title = keep_alphanumeric_space_underscore(title)
            outF.write(str(song_id)+','+str(title)+','+str(album)+','+str(year)+"\n")
            with open(out_textfiles+'/'+'id_'+str(song_id)+'.txt', 'w') as out_songtext:
                for liney in lyrics.split('\n'):
                    liney = re.sub('\\r', '', liney)
                    liney = liney.encode('ascii',errors='ignore')                
                    out_songtext.write(str(liney.decode('utf-8'))+'\n')
            year_dict[song_id] = year
            artist_dict[song_id] = artist_name
            title_dict[song_id] = title  
            album_dict[song_id] = album                 
        outF.close()
        
        #   write tracklist to excel file
        filepath_in = os.path.join(artist_path,artist_name+'_tracklist.txt')
        filepath_out = os.path.join(artist_path,artist_name+'_tracklist.xlsx')
        pd.read_csv(filepath_in, delimiter=",").to_excel(filepath_out, index=False, 
                   header = ['Song_id','Title','Album','Year'])  
        
    save_obj(year_dict, os.path.join(main_path,'Desktop','music_data',data_folder) +'/'+'year_dict')
    save_obj(artist_dict, os.path.join(main_path,'Desktop','music_data',data_folder) +'/'+'artist_dict')            
    save_obj(title_dict, os.path.join(main_path,'Desktop','music_data',data_folder) +'/'+'title_dict')            
    save_obj(album_dict, os.path.join(main_path,'Desktop','music_data',data_folder) +'/'+'album_dict')             

main()