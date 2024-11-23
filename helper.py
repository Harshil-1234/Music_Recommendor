from fer import FER
import cv2
import requests
import numpy as np
from keras import models
# from tensorflow.keras.models import model_from_json


json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = models.model_from_json(model_json)
model.load_weights("model.h5")

haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

def get_emotion(img_path):
    img = cv2.imread(img_path)
    detector = FER()
    emotion, score = detector.top_emotion(img)
    # emotion = predict_emotion(img)
    
    return emotion
    # img = cv2.imread(img_path)
    # if img is None:
    #     return "Image not found or unable to read."

    # emotion = predict_emotion(img)
    # return emotion if emotion else "No face detected."

def predict_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    if(len(faces) == 0):return None
    try:
        for p, q, r, s in faces:
            face_image = gray[q : q + s, p : p + r]
            face_image = cv2.resize(face_image, (48, 48))
            img = extract_features(face_image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

        return prediction_label if prediction_label else None
    except cv2.error:
        return None

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def getToken():
    url = "https://accounts.spotify.com/api/token"

    client_id = "6b79d5ca83484349bac5b36a2a6dc35b"
    client_secret = "75c5f7f6bd004563ac5fac37cdc01b15"

    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }

    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    response = requests.post(url, headers=headers, data=data)

    # print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data.get("access_token")
        return access_token

def getPlaylist(token, emotion):
    playlists = {
        'happy': 'spotify:playlist:37i9dQZF1DWTwbZHrJRIgD',
        'sad': 'spotify:playlist:37i9dQZF1DX3rxVfibe1L0',
        'angry': 'spotify:playlist:37i9dQZF1EIfTmpqlGn32s',
        'surprise': 'spotify:playlist:37i9dQZF1DX2sUQwD7tbmL',
        'neutral': 'spotify:playlist:37i9dQZF1DX0XUfTFmNBRM',
        'disgust': 'spotify:playlist:37i9dQZF1DX3rxVfibe1L0',
        'fear': 'spotify:playlist:37i9dQZF1EIfTmpqlGn32s'
    }
    return playlists.get(emotion)



labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}