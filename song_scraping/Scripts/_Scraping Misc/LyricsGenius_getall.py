#	https://zpitenis.com/blog/post/mining-song-lyrics-from-genius/
#	https://www.r-craft.org/r-news/nas-vs-doom-a-model-based-text-analysis-with-python/

import pandas as pd
import time
import lyricsgenius
client_access_token = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
genius = lyricsgenius.Genius(client_access_token, remove_section_headers=True,
                 skip_non_songs=True, excluded_terms=["Remix", "Live", "Edit", "Mix", "Club"])

#Create list of sample artists
sample_artists = ['Rihanna', 'Justin Timberlake']

#Starting the song search for the artists in question and seconds count
query_number = 0
time1 = time.time()
for artist in sample_artists:
    query_number += 1
    #Empty lists for artist, title, album and lyrics information
    artists = []
    titles = []
    albums = []
    years = []
    lyrics = []
    print('\nQuery number:', query_number)
    #Search for max_songs = n and sort them by popularity
    artist = genius.search_artist(artist, max_songs = 10, sort='popularity')
    songs = artist.songs
    song_number = 0
    #Append all information for each song in the previously created lists
    for song in songs:
        if song is not None:
            song_number += 1
            print('\nSong number:', song_number)
            print('\nNow adding: Artist')
            artists.append(song.artist)
            print('Now adding: Title')
            titles.append(song.title)
            print('Now adding: Album')
            albums.append(song.album)
            print('Now adding: Year')
            years.append(song.year[0:4])
            print('Now adding: Lyrics')
            lyrics.append(song.lyrics)
    time2 = time.time()
    print('\nQuery', query_number, 'finished in', round(time2-time1,2), 'seconds.')

#Create a dataframe for our collected tracklist   
tracklist = pd.DataFrame({'artist':artists, 'title':titles, 'album':albums, 'year':years, 'lyrics':lyrics})   
time3 = time.time()   
print('\nFinal tracklist of', query_number, 'artists finished in', round(time3-time1,2), 'seconds.')
#Save the final tracklist to csv format
tracklist.to_csv('mini_lyrics_corpus.csv', encoding = 'utf-8', index=False)