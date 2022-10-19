import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
import os, os.path
import numpy as np
import sys
import cv2
from preprocessing import process

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

model = keras.models.load_model('model') 

classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    x, y, c = frame.shape
    frame = cv2.flip(frame, 1)
    if cv2.waitKey(1) == ord('q'):
        break
    
    rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed = process([rgbframe]) 

    row = processed[0]

    if len(row):
        row = row[0]
        frame = processed[1][0]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        pred = model.predict(row.reshape(1, 63))
        pred_index = np.argmax(pred)
        cv2.putText(frame, classes[pred_index].upper(), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("Output", frame)

cap.release()
cv2.destroyAllWindows()