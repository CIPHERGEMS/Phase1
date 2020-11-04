# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 13:59:51 2020

@author: Dave
"""

import argparse
import logging

from spotipy.oauth2 import SpotifyClientCredentials
import spotipy

logger = logging.getLogger('examples.artist_discography')
logging.basicConfig(level='INFO')

client_id = 'b38a2f1148af43388452ae0038f3bf5f'
client_secret = '089cc6358c63452e8c44d65267023c75'


def get_args():
    parser = argparse.ArgumentParser(description='Shows albums and tracks for '
                                     'given artist')
    parser.add_argument('-a', '--artist', required=True,
                        help='Name of Artist')
    return parser.parse_args()


def get_artist(name):
    results = sp.search(q='artist:' + name, type='artist')
    items = results['artists']['items']
    if len(items) > 0:
        return items[0]
    else:
        return None


def show_album_tracks(album):
    tracks = []
    results = sp.album_tracks(album['id'])
    tracks.extend(results['items'])
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    for i, track in enumerate(tracks):
        logger.info('%s. %s', i+1, track['name'])


def show_artist_albums(artist):
    albums = []
    results = sp.artist_albums(artist['id'], album_type='album')
    albums.extend(results['items'])
    while results['next']:
        results = sp.next(results)
        albums.extend(results['items'])
    logger.info('Total albums: %s', len(albums))
    unique = set()  # skip duplicate albums
    for album in albums:
        name = album['name'].lower()
        if name not in unique:
            logger.info('ALBUM: %s', name)
            unique.add(name)
            show_album_tracks(album)


def show_artist(artist):
    logger.info('====%s====', artist['name'])
    logger.info('Popularity: %s', artist['popularity'])
    if len(artist['genres']) > 0:
        logger.info('Genres: %s', ','.join(artist['genres']))

def main(artist):
    artist = get_artist(artist)
    show_artist(artist)
    show_artist_albums(artist)


if __name__ == '__main__':
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))
    main('Ghostface Killah')