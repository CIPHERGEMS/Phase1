# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:02:20 2020

@author: Dave
"""

#   https://www.rollingstone.com/music/music-lists/100-greatest-hip-hop-songs-of-all-time-105784/?list_page=1
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

URL = 'https://www.rollingstone.com/music/music-lists/100-greatest-hip-hop-songs-of-all-time-105784/?list_page=2'
page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')
job_elems = soup.find_all('article', class_='c-list__item c-list__item--artist')

outF = open(os.path.join(main_path,'Desktop',"myOutFile.txt"), "w")
for job_elem in job_elems:
   # print(job_elem)
    title_elem = job_elem.find('h3', class_='c-list__title t-bold').text
    title_elem = title_elem.strip()
    title_elem = title_elem.rstrip('\n')
    if not title_elem:
        continue
    outF.write(title_elem.split(',')[0])
    outF.write("\n")
outF.close()
        
