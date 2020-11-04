# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 23:44:55 2020

@author: Dave
"""
#%%
import os
import pickle
import re
import nltk
import pickle
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from uuid import getnode as get_mac
mac = get_mac()

main_path = "/home/dbrowne/"
if mac == 45015790230696:  # dave's laptop
    main_path = "C:/Users/dbrowne/"
if mac == 198112232959511:  # dave's home
    main_path = "C:/Users/dave/"

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)
  
def keep_alpha_space(string):
    return re.sub(r'[^A-Za-z ]+', '', string)
    
def replace_unicode(string):
    return re.sub('\u2005', '', string)

def rm_between_brackets(string):
    return re.sub("[\(\[].*?[\)\]]", " ", string)

def lower_case(string):
    return string.lower()

def rm_stopwords(string):
    # set of stop words
    stop_words = set(stopwords.words('english')) 
    
    # tokens of words  
    word_tokens = word_tokenize(string) 
        
    filtered_sentence = [] 
      
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 

    return " ".join(filtered_sentence)

def lemming(string):
    lemmatizer=WordNetLemmatizer()
    word_tokens = word_tokenize(string) 
    return  "_".join([lemmatizer.lemmatize(word) for word in word_tokens]) 

def clean(string):
    string = rm_between_brackets(string)
    string = keep_alpha_space(string)
    string = lower_case(string)
    string = rm_stopwords(string)
    string = lemming(string)
    return string

data_folder = 'Genius'
user_input = 'don’t play with fire'

search_terms_dict = load_obj(os.path.join(main_path,'Desktop','music_data','search_terms_dict.pkl'))
search_synonyms_dict = load_obj(os.path.join(main_path,'Desktop','music_data','search_synonyms_dict.pkl'))
year_dict = load_obj(os.path.join(main_path,'Desktop','music_data',data_folder,'year_dict.pkl'))
title_dict = load_obj(os.path.join(main_path,'Desktop','music_data',data_folder,'title_dict.pkl'))
artist_dict = load_obj(os.path.join(main_path,'Desktop','music_data',data_folder,'artist_dict.pkl'))
album_dict = load_obj(os.path.join(main_path,'Desktop','music_data',data_folder,'album_dict.pkl'))
ngram_dic_lst = load_obj(os.path.join(main_path,'Desktop','music_data') +'/'+'search_ngram_dic_lst.pkl')
artist_path_lst = [f.path for f in os.scandir(os.path.join(main_path,'Desktop','music_data',data_folder)) if f.is_dir() ]

#   get list of symons
if clean(user_input) not in search_terms_dict:
    search_term_list = [clean(user_input)]
else:
    search_term_list = search_synonyms_dict[search_terms_dict[clean(user_input)]]

#%%
for search_term in search_term_list:
    search_ngrams = len(search_term.split('_'))    
    if 'gram'+str(search_ngrams) not in ngram_dic_lst:
        print('Error.... Max number of terms is 15')
    else:
        ngram_dict = ngram_dic_lst['gram'+str(search_ngrams)]
    
    if search_term in ngram_dict:
        res = ngram_dict[search_term]

        print(search_term+'\n')
        for value in res:
            id_val = int(value.split('_')[0])
            print(title_dict[id_val]+','+artist_dict[id_val]+','+year_dict[id_val]+','+album_dict[id_val]+'\n')