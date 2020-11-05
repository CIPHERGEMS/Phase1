# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 00:12:39 2020

@author: Dave
"""

import os
import pickle
import collections
import pandas as pd
from pathlib import Path,PurePath
from nltk.util import ngrams
from NlpPreprocessing import NlpPreprocessing
from nltk.tokenize import word_tokenize 

home = os.path.expanduser('~')
homebase_path = Path(home)
country_folder = homebase_path / "Desktop" / "USA"
artists_tracks_folder = country_folder / "Got_Tracks"
results_path = homebase_path / "Desktop" / "ngrams_songs_dicts"
max_grams = 15

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

dict_id_year = {}
dict_Artis2Number = {}
dict_id_track = {}
dict_id_album = {}

artist_paths = [f.path for f in os.scandir(artists_tracks_folder) if f.is_file() ]
for artist_num, artist_path in enumerate(artist_paths):
    dict_Artis2Number["Artist" + str(artist_num)] = (Path(artist_path).name).split('.')[0]
    df_artist_tracks = pd.read_csv(artist_path, encoding='latin1')
    df_artist_tracks.columns = ['title','album','year','lyrics','urls']   
    for track_num, ir in enumerate(df_artist_tracks.itertuples()):
        ids = "Artist" + str(artist_num) + "_Track" + str(track_num)
        dict_id_year[ids] = ir[3]
        dict_id_track[ids] = ir[1]
        dict_id_album[ids] = ir[2]        
        NlpPrep = NlpPreprocessing(ir[4])
        clean_track = NlpPrep.start_preprocess()
        word_tokens = word_tokenize(clean_track) 
        for gram in range(1,max_grams+1):
            gram_dict = {}
            # and get a list of all the bi-grams
            esBigrams = ngrams(word_tokens, gram)
            ngramFreq = dict(collections.Counter(esBigrams))
            for k,v in ngramFreq.items():
                gram_dict['_'.join(k)] = v

            out_folder = str(results_path / ids)
            os.makedirs(out_folder, exist_ok=True)
            save_obj(gram_dict, out_folder + '/' + 'gram' + str(gram) + '.pkl') 
            
#%%
dict_year_track = {}
for k,v in dict_id_year.items():
    if v not in dict_year_track:
        dict_year_track[v] = [k]
    else:
        dict_year_track[v] += [k]
save_obj(dict_id_year, results_path / "dict_id_year.pkl")
save_obj(dict_id_track, results_path / 'dict_id_track.pkl')
save_obj(dict_id_album, results_path / 'dict_id_album.pkl')
save_obj(dict_Artis2Number, results_path / 'dict_Artis2Number.pkl')