import cv2
import mediapipe as mp
import os, os.path

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

path = r'C:\Users\jkorr\Downloads\main\images'
apath = r'C:\Users\jkorr\Downloads\main\annotated_images'

adjusted_alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

LANDMARK_LIST = []

# Put image file names into a list
IMAGE_FILES = []
for f in os.listdir(path):
    IMAGE_FILES.append(os.path.join(path, f))

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:

    # Iterate through files in IMAGE_FILES
    for idx, file in enumerate(IMAGE_FILES):
        # Read image, flip it, and convert it (BGR) to RGB
        image = cv2.flip(cv2.imread(file), 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_hand_landmarks:
            continue

        # Copy the image
        annotated_image = image.copy()

        LANDMARK_LIST.append(results.multi_hand_landmarks)

        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            
            # Draw the landmarks onto the annotated image
            """
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            """

        # Add the anotated image to folder
        #cv2.imwrite(
        #    os.path.join(apath, 'annotated_{}'.format(adjusted_alphabet[idx])) + str(idx) + '.png', cv2.flip(annotated_image, 1))
        
        if not results.multi_hand_world_landmarks:
            continue
        
            
        
        