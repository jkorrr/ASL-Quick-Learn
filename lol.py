from logging import logProcesses
from telnetlib import STATUS
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, \
    MaxPool1D, BatchNormalization, LayerNormalization, AveragePooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import os
from matplotlib import pyplot as plt
import cv2


# data processing
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7, static_image_mode=True)
mpDraw = mp.solutions.drawing_utils

classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt = tf.keras.optimizers.Adam(learning_rate=0.001, name = 'Adam')

x_train = np.loadtxt("datasets/train_x.csv", delimiter=",")
# y_train = np.loadtxt("train_y.csv", delimiter=",")
y_train = list(range(24))
y_train.remove(14)
y_train.remove(22)
y_train.remove(23)
y_train = np.array(y_train)
"""
def model(input_dims, l2_lambda):
    inputs = tf.keras.Input(shape=input_dims)
    x = tf.keras.Input(shape=input_dims)
    # x = BatchNormalization()(inputs)
    '''
    for _ in range(2):
        x = Conv1D(kernel_size=11, strides=1, filters=12,
                   use_bias=True, kernel_regularizer=l2(l2_lambda),
                   bias_regularizer=l2(l2_lambda), padding='same',
                   activation='relu')(x)
    x = Flatten()(x)
    '''
    x = Dense(100, activation = 'sigmoid')(x)
    x = Dense(50, activation='sigmoid')(x)
    output = Dense(24, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, output)
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

model_test = model(input_dims=x_train.shape[0], l2_lambda=0.005)

model_test.fit(x_train, y_train, epochs=15)
"""

def model(input_dim, plot):

    # data augmenting

    resize_and_rescale = tf.keras.Sequential([
    layers.Resizing()
    ])

    model = Sequential(
        [
            tf.keras.Input(shape=(input_dim, )),
            Dense(30, activation="relu"),
            Dense(27, activation="relu"),
            Dense(24, activation="linear"),

        ]
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    history = model.fit(x_train, y_train, epochs=600)
    
    if plot:
        plt.plot(history.history['accuracy'])
        plt.title('model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.show()
    return model 

x = model(63, plot=True)

x_test = np.loadtxt("datasets/test_x.csv", delimiter=",").reshape(-1, 63)
print(x_test)

count = 0
for i in range(len(x_test)):
    pred = x.predict(x_test[i].reshape(1,63))
    pred_index = np.argmax(pred)
    print(classes[pred_index])
    src = f"annotated_test_images/annotated_{i}.png"
    img = cv2.imread(src)
    img = cv2.putText(img, classes[pred_index], (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
    2, (0,0,255),2,cv2.LINE_AA)
    if classes[pred_index] == x_test[i]:
        count += 1
    cv2.imwrite(src, img)
print(count/len(x_test))
