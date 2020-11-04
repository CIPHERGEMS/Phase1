# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:31:02 2020

@author: Dave
"""


exp = 'get_metadata'
import os
from uuid import getnode as get_mac
mac = get_mac()

main_path = "/home/dbrowne/"
if mac == 45015790230696:  # dave's laptop
    main_path = "C:/Users/dbrowne/"
if mac == 198112232959511:  # dave's home
    main_path = "C:/Users/dave/"
os.chdir(os.path.join(main_path,'Desktop','scripts_genuis',exp))   
script_name = 'main'

f = open(os.path.join(main_path,'Desktop','scripts_genuis','get_artist_id','artists_to_scrape.txt'), 'r')
for artist in f:
    artist = artist.rstrip('\n')
    if not artist:
        continue
    print(artist)
    art_id, name = artist.split(',')
    dataset_path = os.path.join(main_path,'Desktop','music_data','Genius',name)
    d = str('python ' + script_name+'.py'+ 
                          ' -dataset_path '+str(dataset_path)+ 
                          ' -artist_id ' +str(art_id))
    os.system(d)
