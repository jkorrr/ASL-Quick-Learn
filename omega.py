from tkinter.tix import IMAGE
import cv2
import mediapipe as mp
import os, os.path
import numpy as np
import sys
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

path = r'C:\Users\Abein\Desktop\CalHacks\webpage\train_image'
apath = r'C:\Users\Abein\Desktop\CalHacks\webpage\annotated_train_images\\'
cpath = r'C:\Users\Abein\Desktop\CalHacks\webpage\datasets\train_x.csv'

adjusted_alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']
adjusted_alphabet *= 2
adjusted_alphabet.remove('p')
adjusted_alphabet.remove('x')
adjusted_alphabet.remove('y')
LANDMARK_LIST = []

# Put image file names into a list
IMAGE_FILES = []
li = []
for f in os.listdir(path):
    li.append(os.path.join(path, f))

for idx, image in enumerate(li):
    image = cv2.flip(cv2.imread(image), 1)
    if idx + 8027 == 8041:
        continue
    if idx + 8027 == 8048:
        fmt = '.jpg'
    else:
        fmt = '.PNG'
    for deg in np.arange(10, 30, 10):
        z = np.array(Image.open(f'{path}/IMG_{idx+8027}' + fmt).rotate(30))
        z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{path}/IMG_{8048+idx}' + f'_{deg}'+ '.PNG', z)

for f in os.listdir(path):
    IMAGE_FILES.append(os.path.join(path, f))

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.001) as hands:

    # Iterate through files in IMAGE_FILES
    print(len(IMAGE_FILES))
    for idx, file in enumerate(IMAGE_FILES):
        # Read image, flip it, and convert it (BGR) to RGB
        image = cv2.flip(cv2.imread(file), 1)
        x, y, c = image.shape
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            row = [land.x for land in results.multi_hand_landmarks[0].landmark]
            row += [land.y for land in results.multi_hand_landmarks[0].landmark]
            row += [land.z for land in results.multi_hand_landmarks[0].landmark]
            
            LANDMARK_LIST.append(row)

            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            cv2.imwrite(
                apath + 'annotated_' + str(idx) + '.png', cv2.flip(annotated_image, 1))
            
LANDMARK_LIST = np.array(LANDMARK_LIST)
print(LANDMARK_LIST.shape)
np.savetxt(cpath, LANDMARK_LIST, delimiter=",")