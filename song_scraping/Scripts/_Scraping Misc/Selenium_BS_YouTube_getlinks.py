# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 20:21:09 2020

@author: Dave
"""

#   https://py2py.com/youtube-scraping-using-python-part-1-overview-and-installing-selenium/
#   https://py2py.com/youtube-scraping-using-python-part-2-getting-video-ids/
#   https://py2py.com/private-youtube-scraping-using-python-part-3-scraping-title-and-description/

import time
import pickle
import os
import requests
import csv
from pathlib import Path

from selenium import webdriver
from bs4 import BeautifulSoup as bs

home = os.path.expanduser('~')
homebase_path = Path(home)
os.chdir(homebase_path / "Desktop" / "scraping" )
youtube_path = homebase_path / "Desktop" / "Scrape_YouTube" 
os.makedirs(youtube_path, exist_ok = True)
pickle_folder = youtube_path / "Pickle"
os.makedirs(pickle_folder, exist_ok = True)
 
queries=['Eminem', 'Dr. Dre']#, 'food', 'manufacturing', 'history', 'art and music', 'travel blogs']
#driver_path = 
#%%
base="https://www.youtube.com/results?search_query="
vid_id_dict = {}
for query in queries:
       
    query1=query.replace(" ","+")
     
    link=base+query1
     
    driver = webdriver.Chrome(executable_path = str(homebase_path / "Desktop" / "scraping" / "chromedriver.exe"))
    driver.get(link)
 
    time.sleep(5)
     
    for i in range(0,2):
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(3)
        print(i)
 
    soup=bs(driver.page_source, 'lxml')
    vids = soup.findAll('a',{"class":"yt-simple-endpoint style-scope ytd-video-renderer"})
    print(query)
     
    save_ids=os.path.join(youtube_path,'IDs')
    os.makedirs(save_ids, exist_ok = True)
     
    name=query+".txt"
    save_ids_link=os.path.join(save_ids,name)
     
    f= open(save_ids_link,"a+")
    vid_id_list=[]
 
    for v in vids:
        d=str(v)
        vid_id=d[(d.find("href"))+15:(d.find("id="))-2]
        print(vid_id)
        if (vid_id.find("imple"))==-1:
            vid_id_list.append(vid_id)
             
            f.write(vid_id)
            f.write("\n")
     
    vid_id_dict.update({query:[ids for ids in vid_id_list ]})
    f.close()
vid_ids_dict_pickle_path = os.path.join(str(pickle_folder),"vid_ids_dict.pickle")
pickle_in = open(vid_ids_dict_pickle_path,"wb")
pickle.dump(vid_id_dict,pickle_in)
pickle_in.close()

#%%
#   Get data from pickle list
 
pickle_out = open(os.path.join(str(pickle_folder),"vid_ids_dict.pickle"),"rb")
vid_id_dict=pickle.load(pickle_out)
 
dataset_folder = os.path.join(youtube_path,"Dataset")
os.makedirs(dataset_folder, exist_ok=True)
csv_file_path= os.path.join(youtube_path,'main.csv')
 
base = "https://www.youtube.com/watch?v="
for key, values in vid_id_dict.items():
#    for key in keys:
    query_dataset_folder=os.path.join(dataset_folder,key)
 
    if not os.path.exists(query_dataset_folder):
        os.makedirs(query_dataset_folder)
 
    for VidID in values:            
        r = requests.get(base+VidID)
        soup = bs(r.text)
        name=VidID+".txt"
        save_description_link=os.path.join(query_dataset_folder,name)
 
        f= open(save_description_link,"a+")
 
        for title in soup.findAll('p', attrs={'id': 'eow-description'}):
            description=title.text.strip()
            f.write(description)
            print(description)
        f.close()
 
        for title in soup.findAll('span', attrs={"class": 'watch-title'}):
            vid_title= title.text.strip()
            print(vid_title)
 
        with open(csv_file_path, 'a+') as csvfile:
            fieldnames = ['Video id', 'Title','Description','Category']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'Video id': VidID, 'Title': vid_title, 'Description':description,'Category':key})




