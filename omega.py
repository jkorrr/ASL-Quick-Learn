import cv2
import mediapipe as mp
import os, os.path
import numpy as np
import sys

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

path = r'C:\Users\jkorr\Downloads\main\train_images'
apath = r'C:\Users\jkorr\Downloads\main\annotated_train_images\\'

adjusted_alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']
adjusted_alphabet *= 2
adjusted_alphabet.remove('p')
adjusted_alphabet.remove('x')
adjusted_alphabet.remove('y')
LANDMARK_LIST = []

# Put image file names into a list
IMAGE_FILES = []
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
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_hand_landmarks:
            cv2.namedWindow("Input")
            cv2.imshow('image window', image)
            cv2.waitKey(0)

        row = [land.x for land in results.multi_hand_landmarks[0].landmark]
        row += [land.y for land in results.multi_hand_landmarks[0].landmark]
        row += [land.z for land in results.multi_hand_landmarks[0].landmark]
        
        LANDMARK_LIST.append(row)
        print(adjusted_alphabet[idx])

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
np.savetxt("datasets/train_x.csv", LANDMARK_LIST, delimiter=",")
#np.savetxt("train_y.csv", np.array(adjusted_alphabet), fmt='%s', delimiter=',')

