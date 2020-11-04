# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:44:03 2020

@author: Dave
"""

import argparse
import logging

from spotipy.oauth2 import SpotifyClientCredentials
import spotipy

logger = logging.getLogger('examples.artist_albums')
logging.basicConfig(level='INFO')

client_id = 'b38a2f1148af43388452ae0038f3bf5f'
client_secret = '089cc6358c63452e8c44d65267023c75'
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))


def get_args():
    parser = argparse.ArgumentParser(description='Gets albums from artist')
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

def show_artist_albums(artist):
    albums = []
    results = sp.artist_albums(artist['id'], album_type='album')
    albums.extend(results['items'])
    while results['next']:
        results = sp.next(results)
        albums.extend(results['items'])
    seen = set()  # to avoid dups
    albums.sort(key=lambda album: album['name'].lower())
    for album in albums:
        name = album['name']
        if name not in seen:
            logger.info('ALBUM: %s', name)
            seen.add(name)


def main(artist):
    artist = get_artist(artist)
    if artist:
        show_artist_albums(artist)
    else:
        logger.error("Can't find artist: %s", artist)


if __name__ == '__main__':
    main('Dr. Dre')
