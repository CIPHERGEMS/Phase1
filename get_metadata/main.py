# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 00:05:54 2020

@author: Dave
"""

import requests, re, os, json, pickle, sys
import urllib.request
import urllib.parse 
from bs4 import BeautifulSoup

from search import get_json

######################################################################
########### Main file to run everything #############################
#####################################################################

# Constants
base = "https://api.genius.com"
client_access_token = "********************************"

def connect_lyrics(song_id):
    '''Constructs the path of song lyrics.'''
    url = "songs/{}".format(song_id)
    data = get_json(url)

    # Gets the path of song lyrics
    path = data['response']['song']['path']

    return path


def retrieve_lyrics(song_id):
    '''Retrieves lyrics from html page.'''
    path = connect_lyrics(song_id)

    URL = "http://genius.com" + path
    page = requests.get(URL)

    # Extract the page's HTML as a string
    html = BeautifulSoup(page.text, "html.parser")

    # Scrape the song lyrics from the HTML
    lyrics = html.find("div", class_="lyrics").get_text()
    
    # add by me
    #remove identifiers like chorus, verse, etc
    lyrics = re.sub(r'[\[].*?[\]]', '', lyrics)
    #remove empty lines
    lyrics = os.linesep.join([s for s in lyrics.splitlines() if s])
    return lyrics


def get_song_id(artist_id):
    '''Get all the song id from an artist.'''
    current_page = 1
    next_page = True
    songs = [] # to store final song ids

    while next_page:
        path = "artists/{}/songs/".format(artist_id)
        params = {'page': current_page} # the current page
        data = get_json(path=path, params=params) # get json of songs

        page_songs = data['response']['songs']
        if page_songs:
            # Add all the songs of current page
            songs += page_songs
            # Increment current_page value for next loop
            current_page += 1
            print("Page {} finished scraping".format(current_page))
            # If you don't wanna wait too long to scrape, un-comment this
            #if current_page == 2:
            #    break

        else:
            # If page_songs is empty, quit
            next_page = False

    print("Song id were scraped from {} pages".format(current_page))

    # Get all the song ids, excluding not-primary-artist songs.
    songs = [song["id"] for song in songs
            if song["primary_artist"]["id"] == artist_id]

    return songs


def get_song_information(song_ids):
    '''Retrieve meta data about a song.'''
    # initialize a dictionary.
    song_list = {}
    print("Scraping song information")
    for i, song_id in enumerate(song_ids):
        print("id:" + str(song_id) + " start. ->")

        path = "songs/{}".format(song_id)
        data = get_json(path=path)["response"]["song"]

        song_list.update({
        song_id: {
            "title": data["title"],
            "album": data["album"]["name"] if data["album"] else "<single>",
            "release_date": data["release_date"] if data["release_date"] else "unidentified",
            "featured_artists":
                [feat["name"] if data["featured_artists"] else "" for feat in data["featured_artists"]],
            "producer_artists":
                [feat["name"] if data["producer_artists"] else "" for feat in data["producer_artists"]],
            "writer_artists":
                [feat["name"] if data["writer_artists"] else "" for feat in data["writer_artists"]],
            "genius_track_id": song_id,
            "genius_album_id": data["album"]["id"] if data["album"] else "none"}
        })

        print("-> id:" + str(song_id) + " is finished. \n")
    return song_list

def getList(dict): 
    return dict.keys() 

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def main(dataset_path,artist_id):
    # Grabs all song id's from artist
    songs_ids = []
    for _ in range(5): # loop to make sure we get all th songs
        songs_ids_tp = get_song_id(artist_id)
        print('-----------length of temp dictionary---------------')
        print(len(songs_ids_tp))
        songs_ids += songs_ids_tp
    songs_ids = list(set(songs_ids))
    print('-----------length of final dictionary---------------')
    print(len(songs_ids))

    # Get meta information about songs
    song_meta_dic = get_song_information(songs_ids)

    # Scrape lyrics from the songs
    song_lyric_dic = {}
    for song_id in songs_ids:
        song_lyric_dic[song_id] = retrieve_lyrics(song_id) 
    
    assert getList(song_meta_dic) == getList(song_meta_dic)
    keys_dic = getList(song_meta_dic)
    for key_dic in keys_dic:
        song_meta_dic[key_dic]['lyrics'] = song_lyric_dic[key_dic]
        
    os.makedirs(dataset_path, exist_ok=True)
    save_obj(song_meta_dic, dataset_path+'/'+'metadata')

    
    
wrap = True
if wrap:    
    in_val = sys.argv
    if '-dataset_path' not in in_val or '-artist_id' not in in_val:
        print (__doc__)
        raise NameError('error: input options are not provided')
    else:
        dataset_path = str(in_val[in_val.index('-dataset_path') + 1])   
        artist_id = int(in_val[in_val.index('-artist_id') + 1])
    
else:
    # Network Parameters
    dataset_path = os.path.join('C:/','Users','dave','Desktop','Genius','dr_dre')
    artist_id = 123


main(dataset_path,artist_id)



