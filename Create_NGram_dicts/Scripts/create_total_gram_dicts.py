# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 10:10:05 2020

@author: Dave
"""

import os
import pickle
import re
import collections
import nltk
import pickle
from pathlib import Path,PurePath
from nltk.util import ngrams
from nltk.tokenize import word_tokenize 
from uuid import getnode as get_mac
mac = get_mac()

home = os.path.expanduser('~')
homebase_path = Path(home)
in_folder = homebase_path / "Desktop" / "ngrams_songs_dicts"
results_path = homebase_path / "Desktop" / "ngrams_dicts"
os.makedirs(results_path, exist_ok=True)
max_grams = 15

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

ArtistTrack_paths = [f.path for f in os.scandir(in_folder) if f.is_dir() ]
for gram in range(1,max_grams+1):
    total_gram_dict = {} 
    for ArtistTrack_path in ArtistTrack_paths:  
        ArtistTrack = os.path.split(ArtistTrack_path)[1]
        ArtistTrack_gram_dict = load_obj(os.path.join(ArtistTrack_path, "gram" + str(gram) + ".pkl"))      
        for k,v in ArtistTrack_gram_dict.items():
            if k not in total_gram_dict:
                total_gram_dict[k] = [ArtistTrack + '_'+ str(v)]
            else:
                total_gram_dict[k] += [ArtistTrack + '_'+ str(v)]

    save_obj(total_gram_dict, str(results_path) +'/'+'total_gram'+str(gram))
        