# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 22:31:16 2020

@author: Dave
"""

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import os
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from NlpPreprocessing import NlpPreprocessing
from uuid import getnode as get_mac
mac = get_mac()

main_path = "/home/dbrowne/Desktop/"
if mac == 45015790230696:  # dave's laptop
    main_path = "C:/Users/dbrowne/Desktop/"
if mac == 198112232959511:  # dave's home
    main_path = "C:/Users/dave/Desktop/"

def load_dicts(dic_path):    
    def load_obj(name ):
        with open(name, 'rb') as f:
            return pickle.load(f)
    if os.path.exists(dic_path):
        return load_obj(dic_path)
    else:
        return {}
    
excel_sheet = 'search_terms'
df = pd.read_excel(main_path +'/'+ excel_sheet +'.xlsx')

search_synonyms_dict = load_dicts(os.path.join(main_path,'music_data') +'/'+'search_synonyms_dict.pkl')
search_terms_dict = load_dicts(os.path.join(main_path,'music_data') +'/'+'search_terms_dict.pkl')

#%%
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

for row_num in range(df.shape[0]):
    row = df.iloc[row_num].dropna()
    keys = [NlpPreprocessing(row[0]).start_preprocess()]
    for num_keys in range(1,len(row)):
        keys.append(NlpPreprocessing(row[num_keys].start_preprocess())
    # add to synonyms dict
    if keys[0] not in search_synonyms_dict:
        search_synonyms_dict[keys[0]] = list(set(keys[1:]))
    #   add to term dictionary
    for key in keys:
        if key not in search_terms_dict:
            search_terms_dict[key] = keys[0]

#%%   
save_obj(search_synonyms_dict, os.path.join(main_path,'music_data') +'/'+'search_synonyms_dict.pkl')
save_obj(search_terms_dict, os.path.join(main_path,'music_data') +'/'+'search_terms_dict.pkl')