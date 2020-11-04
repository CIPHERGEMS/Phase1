# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:28:57 2020

@author: Dave
"""

import requests
import concurrent.futures
from bs4 import BeautifulSoup


url = "https://www.lyrics.com/album/3769520/Now-20th-Anniversary%2C-Vol.-2"

# Parse the initial 'album' website 
req = requests.get(url)
html = req.content
soup = BeautifulSoup(html , 'html.parser')

# Find all song's links in 'album' site - these can be found under
# the 'strong' tab, and 'a' tab
links = [tag.find('a')['href'] for tag in soup.find_all('strong')[1:-4]] 

#  or give a list of urls

def getLyrics(url):
    HOST = "https://www.lyrics.com"
    url = HOST + url # songs are found on the HOST website
    # Parse 'song' site
    req = requests.get(url)
    html = req.content
    soup = BeautifulSoup(html , 'html.parser')
    # Obtain the lyrics, which can be found under the 'pre' tab
    return soup.find('pre').text

# Use multi-threading for faster performance - I'll give a small run down:
# max_workers = number of threads - we use an individual thread for each song
with concurrent.futures.ThreadPoolExecutor(max_workers=len(links)) as executor:
    # for every song...
    for j in range(len(links)):
        # run the 'getLyrics' method on an individual thread and get the lyrics
        lyrics = executor.submit(getLyrics, links[j]).result()
        # do whatever with the lyrics ... I simply printed them
        print(lyrics)