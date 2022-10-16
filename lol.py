from logging import logProcesses
from telnetlib import STATUS
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
import keras.layers as layers
import os
from matplotlib import pyplot as plt
import cv2


# data processing
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7, static_image_mode=True)
mpDraw = mp.solutions.drawing_utils

path = r'C:/Users/Abein/Desktop/CalHacks/webpage/datasets/train_x.csv'

classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt = tf.keras.optimizers.Adam(learning_rate=0.001, name = 'Adam')

x_train = np.loadtxt(path, delimiter=",")
# y_train = np.loadtxt("train_y.csv", delimiter=",")
y_train = list(range(24))
temp_list = []
for i in range(24):
    temp_list.append(i)
    temp_list.append(i)
y_train += temp_list
for i in range(3):
    y_train.remove(14)
    y_train.remove(22)
    y_train.remove(23)
y_train.remove(21)
y_train.remove(21)
y_train = np.array(y_train)

def model(input_dim, plot):

    model = keras.Sequential(
        [
            tf.keras.Input(shape=(input_dim, )),
            layers.Dense(90, activation="relu"),
            layers.Dense(50, activation="relu"),
            layers.Dense(30, activation="relu"),
            layers.Dense(24, activation="linear"),

        ]
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    history = model.fit(x_train, y_train, epochs=1000)
    
    if plot:
        plt.plot(history.history['accuracy'])
        plt.title('model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.show()
    return model 

x = model(63, plot=False)
x.save('webpage/model')