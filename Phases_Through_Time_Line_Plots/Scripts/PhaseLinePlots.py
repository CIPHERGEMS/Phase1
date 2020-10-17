# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 20:11:53 2020

@author: Dave
"""


import os
import pickle
import pandas as pd
from pathlib import Path,PurePath

home = os.path.expanduser('~')
data_folder = Path(home)
ngram_folder = data_folder / "Desktop" / "Create_NGram_dicts" / "Results" / "ngrams_dicts"
id_folder = data_folder / "Desktop" / "Create_NGram_dicts" / "Results" / "id_dicts"

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
dict_id_year = load_obj(id_folder / "dict_id_year.pkl")
dict_id_track = load_obj(id_folder / 'dict_id_track.pkl')
dict_id_album = load_obj(id_folder / 'dict_id_album.pkl')
dict_id_artist = load_obj(id_folder / 'dict_Artis2Number.pkl')

dict_gram_1 = load_obj(ngram_folder / 'total_gram1.pkl')
dict_gram_2 = load_obj(ngram_folder / 'total_gram2.pkl')
dict_gram_3 = load_obj(ngram_folder / 'total_gram3.pkl')
dict_gram_4 = load_obj(ngram_folder / 'total_gram4.pkl')
dict_gram_5 = load_obj(ngram_folder / 'total_gram5.pkl')
dict_gram_6 = load_obj(ngram_folder / 'total_gram6.pkl')
dict_gram_7 = load_obj(ngram_folder / 'total_gram7.pkl')
dict_gram_8 = load_obj(ngram_folder / 'total_gram8.pkl')
dict_gram_9 = load_obj(ngram_folder / 'total_gram9.pkl')
dict_gram_10 = load_obj(ngram_folder / 'total_gram10.pkl')
dict_gram_11 = load_obj(ngram_folder / 'total_gram11.pkl')
dict_gram_12 = load_obj(ngram_folder / 'total_gram12.pkl')
dict_gram_13 = load_obj(ngram_folder / 'total_gram13.pkl')
dict_gram_14 = load_obj(ngram_folder / 'total_gram14.pkl')
dict_gram_15 = load_obj(ngram_folder / 'total_gram15.pkl')

#%%
cont = True
while cont:
    #input_search = 'third eye && mafia && mack && gangsta'
    input_search = input("Please enter search phase(s) seperated by '&&':\n")
    search_lst = input_search.split('&&')
    
    date_range_bool = input("Do you want to select date range (y/n):\n")
    if date_range_bool == 'y':
        start_date = input("Please enter start date:\n")
        end_date = input("Please enter end date:\n")
        year_range = [int(start_date),int(end_date)]
    else:
        year_range = []
    
    
    def get_dicts(phase):
        print(phase)
        phase = phase.strip()
        gram_num = len(phase.split(' '))
        phase_search = phase.replace(' ', '_')
        
        if phase_search in eval('dict_gram_'+str(gram_num)):
            finds = eval('dict_gram_'+str(gram_num))[phase_search]
        else:
            finds = []
            
        years, year_str = {}, {}
        if len(finds) == 0:
            return years, year_str
        else:
            for track in finds:
                artist_num, track_num, cnt = track.split('_')
                id_num = artist_num + '_' + track_num
                yr_val = dict_id_year[id_num]
                if isinstance(yr_val, str):   
                    yr_val = yr_val.strip()
                    if yr_val != "UnKnown":
                        yr_val = int(float(yr_val))
                if isinstance(yr_val, float): 
                    yr_val = int(yr_val)
                if yr_val not in years:
                    years[yr_val] = 1      # int(cnt)  
                    year_str[yr_val] = [dict_id_artist[artist_num] +'  '+ dict_id_track[id_num] +'  '+ dict_id_album[id_num] +'  '+ str(cnt)]
                else:
                    years[yr_val] += 1       # int(cnt)
                    year_str[yr_val] += [dict_id_artist[artist_num] +'  '+ dict_id_track[id_num] +'  '+ dict_id_album[id_num] +'  '+ str(cnt)]
        
            return years, year_str
    
    years_dict_lst, year_str_dict_lst, phase_lst = [], [], []
    for phase in search_lst:
        phase_lst.append(phase)
        t_year_dict, t_year_str_dict = get_dicts(phase)
        years_dict_lst.append(t_year_dict)
        year_str_dict_lst.append(t_year_str_dict)
   
    max_yr = 0
    min_yr = 3000
    for t_year_dict in years_dict_lst:
        yrs = [x for x in t_year_dict.keys() if not isinstance(x, str)]
        yrs = [int(i) for i in yrs]
        if max(yrs) > max_yr:
            max_yr = max(yrs)
        if min(yrs) < min_yr and min(yrs) > 1970:
            min_yr = min(yrs)

    val_lst_lst = []
    temp_dict = {}
    max_val = 0
    year_lst = [x for x in range(min_yr,max_yr+1)] + ['UnKnown']
    if len(year_range) == 2:
        year_lst = [x for x in range(year_range[0],year_range[1]+1)] 
    for t_year_dict in years_dict_lst:
        val_list = []
        for yr in year_lst:
            if yr not in t_year_dict:
                temp_dict[str(yr)] = 0
            else:
                temp_dict[str(yr)] = t_year_dict[yr]
        for key in sorted(temp_dict):
            val_list.append(temp_dict[key]) 
        if max(val_list) > max_val:
            max_val = max(val_list)
        val_lst_lst.append(val_list)    
   
    
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    
    col_lst = ['red', 'blue', 'green', 'black', 'pink', 'purple']
    plt.figure(figsize=(20,8)).gca()
    for i in range(len(val_lst_lst)):
        plt.plot(year_lst,val_lst_lst[i], color=col_lst[i], marker='o', label=phase_lst[i])
    plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(int(max_val/10)))
    plt.title('')
    plt.xlabel('Year')
    plt.ylabel('Number of tracks')
    plt.legend(loc="upper right")
    plt.xticks(rotation=90)
    plt.show()   
        
    
    for i,year_str in enumerate(year_str_dict_lst):
        print('Search Phase  ==>   ' + search_lst[i])
        for yr in year_lst:
            if yr in year_str:
                print(str(yr))
                vals = year_str[yr]
                for val in vals:
                    print(val)
                print('\n')
        print('\n')
        
    search_again = input("Do you have another query (y/n):\n")
    if search_again == 'n':
        cont = False
        
print("Goodbye")
#%%
