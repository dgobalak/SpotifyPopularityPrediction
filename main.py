import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

from config import *

client_credentials_manager = SpotifyClientCredentials(
    client_id=CID, client_secret=SECRET)
spotify = spotipy.Spotify(
    client_credentials_manager=client_credentials_manager)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


def find_popularity_comparisons(pop):
    df = pd.read_csv("dataset\data.csv")
    df = df[['name', 'popularity']]
    df_sort = df.iloc[(df['popularity']-pop).abs().argsort()[:5]]
    indices = df_sort.index.tolist()
    similar = [df['name'][x] for x in indices]
    return similar


def get_track_id(artist, search_query):
    searchResults = spotify.search(
        q="artist:" + artist + " track:" + search_query, type="track")
    id = searchResults['tracks']['items'][0]['id']
    pop = searchResults['tracks']['items'][0]['popularity'] 
    year = int(searchResults['tracks']['items'][0]['album']['release_date'][:4])
    return id, year, pop


def get_features(artist, search_query):
    track_id, year, popularity = get_track_id(artist, search_query)
    all_features = spotify.audio_features(track_id)[0]
    features = {}
    features['acousticness'] = all_features['acousticness']
    features['danceability'] = all_features['danceability']
    features['duration_ms'] = all_features['duration_ms']
    features['energy'] = all_features['energy']
    features['instrumentalness'] = all_features['instrumentalness']
    features['key'] = all_features['key']
    features['liveness'] = all_features['liveness']
    features['loudness'] = all_features['loudness']
    features['mode'] = all_features['mode']
    features['speechiness'] = all_features['speechiness']
    features['tempo'] = all_features['tempo']
    features['valence'] = all_features['valence']
    features['popularity'] = popularity
    features['year'] = year
    return features


def predict_pop(artist, search_query):
    data = get_features(artist, search_query)
    # KEYS = ['acousticness', 'danceability', 'duration_ms', 'energy',
    #         'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
    #         'speechiness', 'tempo', 'valence']
    KEYS = ['acousticness', 'energy', 'loudness', 'year']
    values = [data.get(key) for key in KEYS]
    return model.predict([values])[0] * 2


def main():
    print("Spotify Popularity Predictions")
    artist = input('Enter name of artist: ')
    search_query = input('Enter name of song: ')
    pop = round(predict_pop(artist, search_query))
    similar_songs = find_popularity_comparisons(pop)
    
    print(f'The predicted popularity is {pop}/100')
    print("Similar Songs (Based on Popularity) Include:")
    print("- ", end="")
    print("\n- ".join(similar_songs))


if __name__ == '__main__':
    main()