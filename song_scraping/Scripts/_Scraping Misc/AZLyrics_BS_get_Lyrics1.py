# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 01:25:35 2020

@author: Dave
"""

#   https://github.com/AlbertSuarez/azlyrics-scraper


import time
import random
import requests
import re
import unidecode

from bs4 import BeautifulSoup
from stem import Signal
from stem.control import Controller
from fake_useragent import UserAgent

AZ_LYRICS_BASE_URL = 'https://www.azlyrics.com'
SCRAPE_PROXY = 'socks5://127.0.0.1:9050'
SCRAPE_RTD_MINIMUM = 0.1
SCRAPE_RTD_MAXIMUM = 0.5
SCRAPE_RETRIES_AMOUNT = 10
SCRAPE_RTD_ERROR_MINIMUM = 0.5
SCRAPE_RTD_ERROR_MAXIMUM = 1
STR_CLEAN_TIMES = 3
STR_CLEAN_DICT = {
    '\n\n': '\n',
    '\n\r\n': '\n',
    '\r': '',
    '\n': ', ',
    '  ': ' ',
    ' ,': ',',
    ' .': '.',
    ' :': ':',
    ' !': '!',
    ' ?': '?',
    ',,': ',',
    '..': '.',
    '::': ':',
    '!!': '!',
    '??': '?',
    '.,': '.',
    '.:': '.',
    ',.': ',',
    ',:': ',',
    ':,': ':',
    ':.': ':'
}

def clean_url(url_str):
    """
    Cleans a given URL.
    :param url_str: String formatted URL.
    :return: Cleaned string formatted URL.
    """
    url_str = url_str.lower()
    url_str = url_str.strip()
    return url_str


def clean_name(name_str):
    """
    Cleans a given name (song or artist).
    :param name_str: String formatted song.
    :return: Cleaned string formatted song.
    """
    name_str = name_str.lower()
    name_str = name_str.strip()
    name_str = unidecode.unidecode(name_str)
    return name_str


def clean_lyrics(lyrics_str):
    """
    Cleans a given string where song lyrics are.
    :param lyrics_str: String formatted lyrics.
    :return: Cleaned string formatted lyrics.
    """
    lyrics_str = lyrics_str.lower()
    lyrics_str = lyrics_str.strip()
    lyrics_str = unidecode.unidecode(lyrics_str)
    lyrics_str = re.sub('[(\[].*?[)\]]', '', lyrics_str)
    for _ in range(0, STR_CLEAN_TIMES):
        for to_be_replaced, to_replace in STR_CLEAN_DICT.items():
            lyrics_str = lyrics_str.replace(to_be_replaced, to_replace)
    lyrics_str = lyrics_str.strip()
    return lyrics_str

#%%

def _get_html(url):
    """
    Retrieves the HTML content given a Internet accessible URL.
    :param url: URL to retrieve.
    :return: HTML content formatted as String, None if there was an error.
    """
    time.sleep(random.uniform(SCRAPE_RTD_MINIMUM, SCRAPE_RTD_MAXIMUM))  # RTD
    for i in range(0, SCRAPE_RETRIES_AMOUNT):
        try:
            with Controller.from_port(port=9051) as c:
                c.authenticate()
                c.signal(Signal.NEWNYM)
            proxies = {'http': SCRAPE_PROXY, 'https': SCRAPE_PROXY}
            headers = {'User-Agent': UserAgent().random}
            response = requests.get(url, proxies=proxies, headers=headers)
            assert response.ok
            html_content = response.content
            return html_content
        except Exception as e:
            if i == SCRAPE_RETRIES_AMOUNT - 1:
                print(f'Unable to retrieve HTML from {url}: {e}')
            else:
                time.sleep(random.uniform(SCRAPE_RTD_ERROR_MINIMUM, SCRAPE_RTD_ERROR_MAXIMUM))
    return None


def get_artist_url_list(artist_letter):
    """
    Retrieves the AZLyrics website URLs for all the artists given its first character.
    :param artist_letter: First character of an artist.
    :return: List of pairs containing the artist name and its AZLyrics URL.
    """
    artist_url_list = []

    try:
        artist_letter_url = f'{AZ_LYRICS_BASE_URL}/{artist_letter}.html'
        html_content = _get_html(artist_letter_url)
        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')

            column_list = soup.find_all('div', {'class': 'artist-col'})
            for column in column_list:
                for a in column.find_all('a'):
                    artist_name = clean_name(a.text)
                    artist_url = clean_url('{}/{}'.format(AZ_LYRICS_BASE_URL, a['href']))
                    artist_url_list.append((artist_name, artist_url))
    except Exception as e:
        print(f'Error while getting artists from letter {artist_letter}: {e}')

    return artist_url_list


def get_song_url_list(artist_url):
    """
    Retrieves the AZLyrics website URLs for all the songs from an artist AZLyrics URL.
    :param artist_url: AZLyrics URL from a given artist.
    :return: List of pairs containing the song name and its AZLyrics URL.
    """
    song_url_list = []

    try:
        html_content = _get_html(artist_url)
        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')

            list_album_div = soup.find('div', {'id': 'listAlbum'})
            for a in list_album_div.find_all('a'):
                song_name = clean_name(a.text)
                artist_url = clean_url('{}/{}'.format(AZ_LYRICS_BASE_URL, a['href'].replace('../', '')))
                song_url_list.append((song_name, artist_url))
    except Exception as e:
        print(f'Error while getting songs from artist {artist_url}: {e}')

    return song_url_list


def get_song_lyrics(song_url):
    """
    Retrieves and cleans the lyrics of a song given its AZLyrics URL.
    :param song_url: AZLyrics URL from a given song.
    :return: Cleaned and formatted song lyrics.
    """
    song_lyrics = ''

    try:
        html_content = _get_html(song_url)
        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')
            div_list = [div.text for div in soup.find_all('div', {'class': None})]
            song_lyrics = max(div_list, key=len)
            song_lyrics = clean_lyrics(song_lyrics)
    except Exception as e:
        print(f'Error while getting lyrics from song {song_url}: {e}')

    return song_lyrics

#%%
import os
def scrape():
    """
    Processes the main function of the scraper.
    :return: All AZLyrics scraped.
    """
    for artist_letter in AZ_LYRICS_ARTIST_LETTER_LIST:
        # Logging stuff
        print(f'[1] Processing [{artist_letter}] letter...')

        # Downloads file if it is available on Box folder.
        csv_file_name = f'{CSV_FILE}_{artist_letter}.csv'
        print(f'[1] Searching for {csv_file_name} in Box folder...')
        file_id = box_sdk.search_file(BOX_FOLDER_APP_ID, csv_file_name.split('/')[-1])
        if file_id:
            print(f'[1] ---> File found with id [{file_id}]!')
            box_sdk.download_file(file_id, csv_file_name)

        # Iterates over all artists with the given letter.
        print('[1] Scraping artists URLs...')
        artist_url_list = azlyrics.get_artist_url_list(artist_letter)
        print(f'[1] ---> {len(artist_url_list)} artists found with letter [{artist_letter}]')
        for artist_name, artist_url in artist_url_list:
            some_song_added = False
            print(f'[2] Scraping song URLs for {artist_name}...')
            song_url_list = azlyrics.get_song_url_list(artist_url)
            print(f'[2] ---> {len(artist_url_list)} artists found with letter [{artist_letter}]')
            for song_name, song_url in song_url_list:
                print(f'[3] Scraping lyrics for song: [{song_name}]')
                if not csv_parser.exists_song(artist_letter, artist_url, song_url):
                    song_lyrics = azlyrics.get_song_lyrics(song_url)
                    csv_parser.append_to_csv(artist_name, artist_url, song_name, song_url, song_lyrics, artist_letter)
                    some_song_added = True
            # Uploads or updates the CSV on Box per every artist.
            if some_song_added:
                if file_id:
                    file_id = box_sdk.update_file(file_id, csv_file_name)
                else:
                    file_id = box_sdk.upload_file(BOX_FOLDER_APP_ID, csv_file_name)

        # Removes the local version of the CSV for saving storage.
        if os.path.isfile(csv_file_name):
            os.remove(csv_file_name)


if __name__ == '__main__':
    iteration = 1
    while True:
        print(f'Starting iteration number {iteration}...')
        scrape()
        iteration += 1