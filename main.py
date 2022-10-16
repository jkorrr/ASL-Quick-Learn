from flask import Flask, render_template, request, send_file, redirect, url_for
import json
import urllib
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
import os, os.path
import numpy as np
import sys
import cv2



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

x = keras.models.load_model('model') 

classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods=['POST'])
def test():
    output = request.get_json()
    output = json.loads(output)
    response = urllib.request.urlopen(output)
    with open('image.jpg', 'wb') as f:
        f.write(response.file.read())

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.001) as hands:
        image = cv2.imread('image.jpg', 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            row = [land.x for land in results.multi_hand_landmarks[0].landmark]
            row += [land.y for land in results.multi_hand_landmarks[0].landmark]
            row += [land.z for land in results.multi_hand_landmarks[0].landmark]

            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            cv2.imwrite(
                'static/annotated.png', cv2.flip(annotated_image, 1))

    pred = x.predict(np.array(row).reshape(1,42))
    pred_index = np.argmax(pred)
    img = cv2.imread("static/annotated.png")
    img = cv2.putText(img, classes[pred_index], (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
    2, (0,0,255),2,cv2.LINE_AA)
    cv2.imwrite("static/annotated.png", img)

    return render_template('index.html')


