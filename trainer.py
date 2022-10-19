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
from preprocessing import process

# Loading paths for data
path = r'C:\Users\Abein\Desktop\CalHacks\ASL QuickLearn\train_image'
ypath = r'C:/Users/Abein/Desktop/CalHacks/ASL QuickLearn/datasets/train_y.csv'

classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

# Data processing
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7, static_image_mode=True)
mpDraw = mp.solutions.drawing_utils

# Loading data from paths
train_images = []
for f in os.listdir(path):
    train_images.append(cv2.imread(os.path.join(path, f)))
x_processed = process(train_images)

x_train = x_processed[0]
y_train = np.loadtxt(ypath, delimiter=",")

# Creating neural network model
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt = tf.keras.optimizers.Adam(learning_rate=0.001, name = 'Adam')

def model(train_x, train_y, input_dim, plot, epochs):

    model = keras.Sequential(
        [
            tf.keras.Input(shape=(input_dim, )),
            layers.Dense(30, activation="relu"),
            layers.Dense(27, activation="relu"),
            layers.Dense(24, activation="linear")

        ]
    )

    model.compile(
        loss=loss,
        optimizer=opt,
        metrics=['accuracy']
    )
    history = model.fit(train_x, train_y, epochs=epochs)
    
    # Plot training accuracy vs epochs
    if plot:
        plt.plot(history.history['accuracy'])
        plt.title('model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.show()
    return model 

x = model(x_train, y_train, 63, plot=False, epochs=200)
x.save('webpage/model')