# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 21:22:44 2020

@author: dbrowne
"""

#   https://github.com/elmoiv/azapi

from math import floor
  
def jaro_distance(s1, s2): 
    if (s1 == s2): 
        return 1.0
  
    len1, len2 = len(s1), len(s2)
    max_dist = floor(max(len1, len2) / 2) - 1
    match = 0
    hash_s1, hash_s2 = [0] * len(s1), [0] * len(s2)
  
    for i in range(len1):
        for j in range(max(0, i - max_dist),  
                       min(len2, i + max_dist + 1)):
            if (s1[i] == s2[j] and hash_s2[j] == 0):
                hash_s1[i], hash_s2[j] = 1, 1
                match += 1
                break

    if (match == 0): 
        return 0.0

    t = 0
    point = 0
  
    for i in range(len1): 
        if (hash_s1[i]): 
            while (hash_s2[point] == 0): 
                point += 1
  
            if (s1[i] != s2[point]): 
                point += 1
                t += 1
    t = t//2

    return (match/ len1 + match / len2 + 
            (match - t + 1) / match)/ 3.0


import requests, random

userAgents = '''Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36
Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.1 Safari/537.36
Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_8) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.11 Safari/535.19
Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.66 Safari/535.11
Mozilla/5.0 (X11; U; Linux x86_64; en-US) AppleWebKit/532.2 (KHTML, like Gecko) Chrome/4.0.221.3 Safari/532.2
Mozilla/5.0 (X11; U; Linux i686; en-US) AppleWebKit/532.2 (KHTML, like Gecko) Chrome/4.0.221.0 Safari/532.2
Mozilla/5.0 (Windows; U; Windows NT 6.0; en-US) AppleWebKit/532.1 (KHTML, like Gecko) Chrome/4.0.220.1 Safari/532.1
Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/532.1 (KHTML, like Gecko) Chrome/4.0.219.6 Safari/532.1
Mozilla/5.0 (Windows; U; Windows NT 5.2; en-US) AppleWebKit/532.1 (KHTML, like Gecko) Chrome/4.0.219.5 Safari/532.1
Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/532.1 (KHTML, like Gecko) Chrome/4.0.219.5 Safari/532.1
Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/532.1 (KHTML, like Gecko) Chrome/4.0.219.4 Safari/532.1
Mozilla/5.0 (X11; U; Linux x86_64; en-US) AppleWebKit/532.1 (KHTML, like Gecko) Chrome/4.0.219.3 Safari/532.1
Mozilla/5.0 (Windows; U; Windows NT 5.2; en-US) AppleWebKit/532.1 (KHTML, like Gecko) Chrome/4.0.219.3 Safari/532.1
Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/532.1 (KHTML, like Gecko) Chrome/4.0.219.3 Safari/532.1
Mozilla/5.0 (X11; U; Linux i686; en-US) AppleWebKit/532.0 (KHTML, like Gecko) Chrome/3.0.197.0 Safari/532.0
Mozilla/5.0 (X11; U; Linux i686 (x86_64); en-US) AppleWebKit/532.0 (KHTML, like Gecko) Chrome/3.0.197.0 Safari/532.0
Mozilla/5.0 (Windows; U; Windows NT 6.0; en-US) AppleWebKit/530.5 (KHTML, like Gecko) Chrome/2.0.172.23 Safari/530.5
Mozilla/5.0 (Windows; U; Windows NT 6.0; en-US) AppleWebKit/530.5 (KHTML, like Gecko) Chrome/2.0.172.2 Safari/530.5
Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/530.5 (KHTML, like Gecko) Chrome/2.0.172.2 Safari/530.5
Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/530.4 (KHTML, like Gecko) Chrome/2.0.172.0 Safari/530.4
Mozilla/5.0 (Windows; U; Windows NT 5.2; eu) AppleWebKit/530.4 (KHTML, like Gecko) Chrome/2.0.172.0 Safari/530.4
Mozilla/5.0 (Windows; U; Windows NT 5.2; en-US) AppleWebKit/530.4 (KHTML, like Gecko) Chrome/2.0.172.0 Safari/530.4
Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/530.5 (KHTML, like Gecko) Chrome/2.0.172.0 Safari/530.5
Mozilla/5.0 (Windows; U; Windows NT 6.0; en-US) AppleWebKit/530.4 (KHTML, like Gecko) Chrome/2.0.171.0 Safari/530.4
Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/530.1 (KHTML, like Gecko) Chrome/2.0.170.0 Safari/530.1
Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/530.1 (KHTML, like Gecko) Chrome/2.0.169.0 Safari/530.1
Mozilla/5.0 (Windows; U; Windows NT 6.0; en-US) AppleWebKit/530.1 (KHTML, like Gecko) Chrome/2.0.168.0 Safari/530.1
Mozilla/5.0 (Windows; U; Windows NT 6.0; en-US) AppleWebKit/530.1 (KHTML, like Gecko) Chrome/2.0.164.0 Safari/530.1
Mozilla/5.0 (Windows; U; Windows NT 6.0; en-US) AppleWebKit/530.0 (KHTML, like Gecko) Chrome/2.0.162.0 Safari/530.0
Mozilla/5.0 (Windows; U; Windows NT 6.0; en-US) AppleWebKit/530.0 (KHTML, like Gecko) Chrome/2.0.160.0 Safari/530.0
Mozilla/5.0 (Windows; U; Windows NT 6.0; en-US) AppleWebKit/528.10 (KHTML, like Gecko) Chrome/2.0.157.2 Safari/528.10
Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/528.10 (KHTML, like Gecko) Chrome/2.0.157.2 Safari/528.10
Mozilla/5.0 (Windows; U; Windows NT 6.0; en-US) AppleWebKit/528.11 (KHTML, like Gecko) Chrome/2.0.157.0 Safari/528.11
Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/528.9 (KHTML, like Gecko) Chrome/2.0.157.0 Safari/528.9
Mozilla/5.0 (Linux; U; en-US) AppleWebKit/525.13 (KHTML, like Gecko) Chrome/0.2.149.27 Safari/525.13
Mozilla/5.0 (Macintosh; U; Mac OS X 10_6_1; en-US) AppleWebKit/530.5 (KHTML, like Gecko) Chrome/ Safari/530.5
Mozilla/5.0 (Macintosh; U; Mac OS X 10_5_7; en-US) AppleWebKit/530.5 (KHTML, like Gecko) Chrome/ Safari/530.5
Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_5_6; en-US) AppleWebKit/530.9 (KHTML, like Gecko) Chrome/ Safari/530.9
Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_5_6; en-US) AppleWebKit/530.6 (KHTML, like Gecko) Chrome/ Safari/530.6
Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_5_6; en-US) AppleWebKit/530.5 (KHTML, like Gecko) Chrome/ Safari/530.5'''

class Requester():
    USER_AGENTS = userAgents.split('\n')
    
    # Inspired from: https://github.com/brianchesley/Lyrics/blob/master/lyrics_data_scrape.py
    def get(self, url, _proxies={}):
        return requests.get(url, headers={'User-Agent': random.choice(self.USER_AGENTS)}, proxies=_proxies)
    
    def head(self, url, _proxies={}):
        return requests.head(url, headers={'User-Agent': random.choice(self.USER_AGENTS)}, proxies=_proxies)


import bs4, re, time, os
from urllib.parse import quote

letters = 'abcdefghijklmnopqrstuvwxyz0123456789'

def htmlFind(page):
    # v3.0
    # Changed page.text -> page.content.decode() to support variant unicodes
    soup = bs4.BeautifulSoup(
                        page.content.decode(),
                        "html.parser"
                        )
    return soup.find

def htmlFindAll(page):
    # v3.0
    # Changed page.text -> page.content.decode() to support variant unicodes
    soup = bs4.BeautifulSoup(
                        page.content.decode(),
                        "html.parser"
                        )
    return soup.findAll

def filtr(inpt, isFile=False):
    if isFile:
        return ''.join(i for i in inpt if i not in r'<>:"/\|?*')
    return ''.join(i.lower() for i in inpt if i.lower() in letters)

def NormalGet(artist='', title='', _type=0):
    art, tit = filtr(artist), filtr(title)
    if _type:
        return 'https://www.azlyrics.com/{}/{}.html'.format(art[0], art)
    return 'https://www.azlyrics.com/lyrics/{}/{}.html'.format(art, tit)

def GoogleGet(srch_eng, acc, get_func, artist='', title='', _type=0):
    # Encode artist and title to avoid url encoding errors
    data = artist + ' ' * (title != '' and artist != '') + title
    encoded_data = quote(data.replace(' ', '+'))

    # Perform a search (for accuracy) [Custom search engine]
    search_engines = {
        'google': 'https://www.google.com/search?q=',
        'duckduckgo': 'https://duckduckgo.com/html/?q='
    }

    slctd_srch_engn = 'google'
    if srch_eng in search_engines:
        slctd_srch_engn = srch_eng

    google_page = get_func('{}{}+site%3Aazlyrics.com'.format(
                                    search_engines[slctd_srch_engn],
                                    encoded_data
                                    )
                            )
    
    # Choose between lyrics or song according to function used
    regex = [
        r'(azlyrics\.com\/lyrics\/(\w+)\/(\w+).html)',
        r'(azlyrics\.com\/[a-z0-9]+\/(\w+).html)'
    ]
    
    # ex result: [('azlyrics.com/t/taylorswift.html', 'taylorswift')]
    # result[0][0] = 'azlyrics.com/t/taylorswift.html'
    results = re.findall(
                        regex[_type],
                        google_page.text
                        )

    if len(results):
        # calculate jaro similarity for artist and title
        jaro_artist = 1.0
        jaro_title = 1.0
        
        if artist:
            jaro_artist = jaro_distance(
                                        artist.replace(' ', ''),
                                        results[0][1]
                                        )
        if title:
            jaro_title = jaro_distance(
                                        title.replace(' ', ''),
                                        results[0][2]
                                        )
        
        if jaro_artist >= acc and jaro_title >= acc:
            return 'https://www.' + results[0][0]
        else:
            print('Similarity <', acc)
    else:
        print(srch_eng.title(), 'found nothing!')
    
    return 0

def ParseLyric(page):
    divs = htmlFindAll(page)('div')
    for div in divs:
        # Lyrics div has no class
        # So we fast check if there is a class or not
        try:
            div['class']
            continue
        except:
            pass
        # Almost all lyrics have more than one <br> tag
        # v3.0: some songs are too short like: Animals - Matin Garrix
        found = div.find_all('br')
        if len(found):
            # Removing unnecessary lines
            return div.text[2:-1]

def ParseSongs(page):
    songs = {}
    Parent = htmlFind(page)('div', {'id':'listAlbum'})
    if Parent:
        Raw_Data = Parent.findChildren()

        curType, curName, curYear = '', '', ''

        for elmnt in Raw_Data:
            
            # v3.0.3: Removed break after script due to google ads inside listAlbum
            # is using script tag, which results in not all songs retrieved
            #if elmnt.name == 'script':
            #    break
            
            # album info are inside divs
            if elmnt.name == 'div':
                if elmnt.text == 'other songs:':
                    curType, curName, curYear = 'Others', '', ''
                else:
                    # Separating to (album, name, year)
                    rgx = re.findall(r'(.*):\s"(.*)"\s\(([0-9]+)\)', elmnt.text)
                    if rgx:
                        curType, curName, curYear = rgx[0]
            if elmnt.name == 'a':
                songs[elmnt.text] = {
                    'year': curYear,
                    'album': curName,
                    'type': curType,
                    # Azlyrics puts hrefs with/without base url
                    'url': 'http://www.azlyrics.com' + elmnt['href'][2:] \
                            if elmnt['href'][:2] == '..' else elmnt['href']
                }
    # v 3.0
    # Some artists have no albums, so we cover this
    else:
        for div in htmlFindAll(page)('div', {'class':'listalbum-item'}):
            a = div.find('a')
            songs[a.text] = {
                'year': '',
                'album': '',
                'type': '',
                # v3.0.1: fix relative urls -> absolute url
                'url': 'http://www.azlyrics.com' + a['href'][2:] \
                        if a['href'][:2] == '..' else a['href']
                }
    return songs


class AZlyrics(Requester):
    """
    Fast and Secure API for AZLyrics.com
    Attributes:
        title (str): song title
        artist (str): singer name
        search_engine (str): search engine used to assist scraping lyrics
            - currently available: 'google', 'duckduckgo'
        accuracy (float): used to determine accuracy via jaro algorithm
        proxies (dict): if you want to use proxy while connecting to AZLyrics.com
    """
    
    def __init__(self, search_engine='', accuracy=0.6, proxies={}):
        self.title = ''
        self.artist = ''
        self.search_engine = search_engine
        
        self.accuracy = accuracy
        if not 0 < accuracy <= 1:
            self.accuracy = 0.6
        
        self.proxies = proxies

        self.lyrics_history = []
        self.lyrics = ''
        self.songs = {}

    def getLyrics(self, url=None, ext='txt', save=False, path='', sleep=10):
        """
        Reterive Lyrics for a given song details
        
        Parameters: 
            url (str): url of the song's Azlyrics page. 
            ext (str): extension of the lyrics saved file, default is ".txt".
            save (bool): allow to or not to save lyrics in a file.
            sleep (float): cooldown before next request.  
        
        Returns:
            lyrics (str): Lyrics of the detected song
        """

        if not self.artist + self.title:
            raise ValueError("Both artist and title can't be empty!")
        
        # Best cooldown is 5 sec
        time.sleep(sleep)

        link = url

        if not url:
            if self.search_engine:
                # If user can't remember the artist,
                # he can search by title only
                
                # Get AZlyrics url via Google Search
                link = GoogleGet(
                            self.search_engine,
                            self.accuracy,
                            self.get,
                            self.artist,
                            self.title,
                            0)
                if not link:
                    return 0
            else:
                # Sometimes search engines block you
                # If happened use the normal get method
                link = NormalGet(
                            self.artist,
                            self.title,
                            0)

        page = self.get(link)
        if page.status_code != 200:
            print('Error 404!')
            return 1

        # Getting Basic metadata from azlyrics
        metadata = [elm.text for elm in htmlFindAll(page)('b')]
        
        # v3.0.4: Update title and artist attributes with exact names
        self.artist = filtr(metadata[0][:-7], True)
        self.title = filtr(metadata[1][1:-1], True)

        lyrics = ParseLyric(page)
        if lyrics is not None:
            self.lyrics = lyrics.strip()

        # Saving Lyrics
        if lyrics:
            if save:
                # v3.0.2: Adding custom path
                p = os.path.join(
                                path,
                                '{} - {}.{}'.format(
                                                self.title.title(),
                                                self.artist.title(),
                                                ext
                                                )
                                )
                
                with open(p, 'w', encoding='utf-8') as f:
                    f.write(lyrics.strip())
            
            # Store lyrics for later usage
            self.lyrics_history.append(self.lyrics)
            return self.lyrics

        self.lyrics = 'No lyrics found :('
        return 2

    def getSongs(self, sleep=10):
        """
        Reterive a dictionary of songs with their links
        Parameters:
            sleep (float): cooldown before next request.  
        
        Returns:
            dict: dictionary of songs with their links
        """

        if not self.artist:
            raise Exception("Artist can't be empty!")
        
        # Best cooldown is 5 sec
        time.sleep(sleep)
        
        if self.search_engine:
            link = GoogleGet(
                        self.search_engine,
                        self.accuracy,
                        self.get,
                        self.artist,
                        '',
                        1)
            if not link:
                return {}
        else:
            link = NormalGet(
                        self.artist,
                        '',
                        1)
        
        albums_page = self.get(link)
        if albums_page.status_code != 200:
            print('Error 404!')
            return {}
        
        # Store songs for later usage
        self.songs = ParseSongs(albums_page)
        return self.songs
            
#%%
import os
import xlrd
import pandas as pd
from pathlib import Path

home = os.path.expanduser('~')
homebase_path = Path(home)
scraping_folder = homebase_path / "Desktop" / "song_scraping"
os.chdir(scraping_folder)

for dirname, _, filenames in os.walk(scraping_folder / "Artist_to_scrape"):
    for filename in filenames:
        workbook = xlrd.open_workbook(os.path.join(dirname, filename))
        sheet = workbook.sheet_by_index(0)
        for rowx in range(sheet.nrows):
            current_artist = sheet.row_values(rowx)[0]
            print(current_artist)
            #Empty lists for artist, title, album and lyrics information
            titles = []
            albums = []
            years = []
            lyrics = []
            urls = []
            #create output folder
            out_folder = scraping_folder / "all_tracks_from_sites_with_duplicates" / filename.split('_')[0]
            os.makedirs(out_folder, exist_ok=True)
            #get songs and meta data from website
            api = AZlyrics()
            api.artist = current_artist
            songs = api.getSongs()
            # if website has no songs for artist skip and append "0" to end to identify
            if len(songs) == 0:
                outF = open(os.path.join(scraping_folder, "all_tracks_from_sites_with_duplicates", filename.split('_')[0], current_artist+'__azlyrics.txt'), "w")
                outF.write('')
                outF.close()
            # write songs and meta to dict to sort and then to txt file
            else:
                for song, meta in songs.items():
                    titles.append(song)
                    albums.append(meta['album'])
                    years.append(meta['year'])
                    urls.append(meta['url'])
                    lyrics.append(api.getLyrics(meta['url']))

                #Create a dataframe for our collected tracklist   
                tracklist = pd.DataFrame({'title':titles, 'album':albums, 'year':years, 'lyrics':lyrics, 'urls':urls})     
                #Save the final tracklist to csv format
                tracklist.to_csv(os.path.join(scraping_folder, "all_tracks_from_sites_with_duplicates", filename.split('_')[0], current_artist+'__azlyrics.csv'), encoding = 'utf-8', index=False)