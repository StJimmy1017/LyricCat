import secrets
import numpy as np
import requests
import gensim.downloader as api
from numpy import dot
from numpy.linalg import norm

model = api.load("glove-wiki-gigaword-100")
key = secrets.API
BASE_URL = 'http://api.musixmatch.com/ws/1.1/'

def get_song_names(query):
    params = {
        'apikey': key,
        'q_lyrics': query,
        's_track_rating': 'desc',
        'f_lyrics_language': 'en',
        'format': 'json'
    }
    response = requests.get(BASE_URL + 'track.search', params=params)
    data = response.json()
    if 'message' in data and 'body' in data['message'] and 'track_list' in data['message']['body']:
        return data['message']['body']['track_list']
    else:
        return []

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def calculate_embedding_glove(text):
    words = text.lower().split()
    embedding = np.zeros_like(model['a'])
    num_words = 0
    for word in words:
        if word in model:
            embedding += model[word]
            num_words += 1
    if num_words > 0:
        return embedding / num_words
    return None
