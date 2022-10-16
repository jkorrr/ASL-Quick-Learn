from telnetlib import STATUS
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
from tensorflow.keras import load_model


# data processing
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7, static_image_mode=True)
mpDraw = mp.solutions.drawing_utils

classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

model = load_model('mp_hand_gesture')
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt = tf.keras.optimizers.Adam(learning_rate=0.001, name = 'Adam')
# data =  define dataset

class CNN:
    def __init__(self, input_dims):
        self.conv1 = keras.layers.Conv2d(1, 5, 5, 1)
        self.maxpool1 = keras.layers.MaxPool2d((2,2), stride=2)
        self.conv2 = keras.layers.Conv2d(5, 16, 5, 1)
        self.maxpool2 = keras.layers.MaxPool2d((2,2), stride=2)
        self.dropout = keras.layers.Dropout(0.5)
        self.linear1 = keras.layers.Linear(1500, 120)
        self.linear2 = keras.layers.Linear(120, 84)
        self.linear3 = keras.layers.Linear(84, 10) 

    def forward_pass(self, x):
        x = self.conv1(x)
        x = tf.nn.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.maxpool2(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)

        return x

