# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 22:06:03 2020

@author: Dave
"""

#%%
import requests
from bs4 import BeautifulSoup
import os
from uuid import getnode as get_mac
mac = get_mac()

main_path = "/home/dbrowne/"
if mac == 45015790230696:  # dave's laptop
    main_path = "C:/Users/dbrowne/"
if mac == 198112232959511:  # dave's home
    main_path = "C:/Users/dave/"

URL = 'https://rateyourmusic.com/list/Matica/151_greatest_hip_hop_songs_according_to_the_source/1/'
page = requests.get(URL)
page = main_path+'/'+'Desktop/scripts_genuis/get_metadata/source_151.html'
f = open(page) 
soup = BeautifulSoup(f)
f.close()
job_elems = soup.find_all('a', class_='list_artist')

outF = open(os.path.join(main_path,'Desktop',"myOutFile.txt"), "w")
for job_elem in job_elems:
    artist = job_elem.get('href')
    artist = artist.split('/')[-1]
    artist = artist.strip()
    artist = artist.rstrip('\n')
    if not artist:
        continue
    outF.write(artist)
    outF.write("\n")
outF.close()
        