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

path = r'C:\Users\Abein\Desktop\CalHacks\ASL QuickLearn\train_image'
apath = r'C:\Users\Abein\Desktop\CalHacks\ASL QuickLearn\annotated_train_images\\'
xpath = r'C:\Users\Abein\Desktop\CalHacks\ASL QuickLearn\datasets\train_x.csv'

def process(image_files):

    landmark_list = []
    annotated_images = []

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.001) as hands:

        # Iterate through files in IMAGE_FILES
        # print(len(image_files))
        for file in image_files:
            # Read image, flip in, and convert it (BGR) to RGB
            image = cv2.flip(file, 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                
                # Save landmark values in LANDMARK_LIST
                row = [land.x for land in results.multi_hand_landmarks[0].landmark]
                row += [land.y for land in results.multi_hand_landmarks[0].landmark]
                row += [land.z for land in results.multi_hand_landmarks[0].landmark]
                
                landmark_list.append(row)

                # Annotate image with landmarks
                annotated_image = image.copy()
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                annotated_images.append(annotated_image)
                
    landmark_list = np.array(landmark_list)

    return landmark_list, annotated_images