from flask import Flask, render_template, request, jsonify, redirect, url_for
import cv2
import base64
import numpy as np
from helper import get_emotion, getToken, getPlaylist

from deepface import DeepFace
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('dashboard.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/feature')
def feature():
    return render_template('feature.html')

@app.route('/try', methods=['GET', 'POST'])
def emotunes():
    if request.method == 'POST':
        try:
            data = request.get_json()
            img_data = data['image']
            img_data = base64.b64decode(img_data.split(',')[1])
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            cv2.imwrite('static/images/captured_image.png', img)

            emotion = get_emotion(img_path='static/images/captured_image.png')

            return jsonify({'redirect': '/playlist', 'emotion': emotion})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return render_template('emotunes.html')

@app.route('/playlist', methods=['GET'])
def playlist():
    emotion = request.args.get('emotion')
    token = getToken()
    playlist_uri = getPlaylist(token, emotion)
    print(f"Emotion: {emotion}, Playlist URI: {playlist_uri}")  # Debugging
    return render_template('playlist.html', image='static/images/captured_image.png', emotion=emotion, playlist_uri=playlist_uri)
    # return render_template('playlist.html', image='./sudo.jpeg', emotion=emotion, playlist_uri=playlist_uri)

# FOR SONGS FROM DATASET

# @app.route('/playlist', methods=['GET'])
# def get_songs():
#     try:
#         emotion = request.args.get('emotion')

#         # Load song data
#         df = pd.read_csv('SpotifySongs.csv')
#         df.drop(columns=['Duration_ms', 'Popularity'], inplace=True)
#         df.isnull().sum()
#         df.dropna(inplace=True)
#         df = df.drop_duplicates()

#         # Scale song features
#         scaler = MinMaxScaler()
#         df_scaled = scaler.fit_transform(df.iloc[:, 2:])
#         df_scaled = pd.DataFrame(df_scaled, columns=df.columns[2:])
#         df_scaled = pd.concat([df.iloc[:, :2], df_scaled], axis=1)
#         df_scaled = df_scaled.drop_duplicates()
#         df_scaled.dropna(inplace=True)

#         # Define song features for each emotion
#         happy_song_features = [[0.8, 0.7, 0.0, 0.7, 0.0, 0.5, 0.1, 0.1, 0.2, 0.5, 120]]
#         sad_song_features = [[0.4, 0.4, 0.5, 0.4, 0.5, 0.3, 0.05, 0.05, 0.05, 0.3, 100]]
#         angry_song_features = [[0.5, 0.9, 0.0, 0.8, 0.5, 0.5, 0.15, 0.35, 0.15, 0.4, 140]]
#         neutral_song_features = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.5, 110]]

#         # Transform features based on detected emotion
#         if emotion == 'happy':
#             x = scaler.transform(happy_song_features)
#         elif emotion == 'sad':
#             x = scaler.transform(sad_song_features)
#         elif emotion == 'angry':
#             x = scaler.transform(angry_song_features)
#         else:
#             x = scaler.transform(neutral_song_features)

#         cosine_similarities = cosine_similarity(x, df_scaled.drop(columns=['SongName', 'ArtistName']))
#         similar_song_indices = cosine_similarities.argsort()[0][::-1]

#         # Get top 50 similar songs
#         similar_songs = df_scaled.iloc[similar_song_indices[:50]]['SongName'].tolist()

#         return jsonify({'emotion': emotion, 'similar_songs': similar_songs})

#     except Exception as e:
#         return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=8000)