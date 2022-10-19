from flask import Flask, render_template, request
from preprocessing import process
import json
import urllib
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
import os, os.path
import numpy as np
import sys
import cv2
from preprocessing import process

i = 0

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

model = keras.models.load_model('model') 

classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods=['POST'])
def test():
    global i
    output = request.get_json()
    output = json.loads(output)
    response = urllib.request.urlopen(output)
    with open('image.jpg', 'wb') as f:
        f.write(response.file.read())

    image = cv2.imread("image.jpg", 1)

    processed = process([image])
    row = processed[0]
    if len(row):
        annotated_image = processed[1][0]
        cv2.imwrite(f'static/annotated{i}.png', annotated_image)

        pred = model.predict(row.reshape(1,63))
        pred_index = np.argmax(pred)
        img = cv2.imread(f"static/annotated{i}.png")
        img = cv2.putText(img, classes[pred_index].upper(), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),2,cv2.LINE_AA)

    cv2.imwrite(f"static/annotated{i}.png", img)
    if i > 0:
        os.remove(f"static/annotated{i - 1}.png")
    i += 1

    return render_template('index.html')


