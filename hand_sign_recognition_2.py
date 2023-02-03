import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import os
import numpy as np
import mediapipe as mp

names=[]
for foldername in os.listdir(r'D:\programming\SE_project_files\asl_images'):
    names.append(foldername)

model=keras.models.load_model('hand_recognise_model_own90_100_.h5')

cap=cv.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mp_draw=mp.solutions.drawing_utils

def find_hand():
    while True:
        success, img = cap.read()
        top = set()
        bottom = set()
        left = set()
        right = set()
        # converting to RGB:
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # processing entire image to detect and track the palm:
        results = hands.process(img_rgb)

        # results.multi_hand_landmarks has the landmarks for every hand it detects in frame ,now we are drawing those landmarks and connections on image:
        if (results.multi_hand_landmarks):
            for hand in results.multi_hand_landmarks:
                # to do ---> improve the space complexity:

                for id, lm in enumerate(hand.landmark):
                    # id is basically the landmark number and lm is the x,y coordinate of that landmark.
                    # lm values are not in pixels , its as ratio, so finding pixel values:
                    height, width, channels = img.shape
                    x, y = int(lm.x * width), int(lm.y * height)

                    top.add(y + 20)
                    bottom.add(y - 20)
                    left.add(x - 20)
                    right.add(x + 20)

                    mp_draw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)


        if (len(top) != 0 and len(bottom) != 0 and len(left) != 0 and len(right) != 0):
            img_normal = img
            return  [img_normal,img_normal[min(bottom):max(top), min(left):max(right)]]
        else:
            img_normal = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
            return [img_normal,[]]

feature=np.zeros((1,90,90,3))
feature.astype(np.uint8)

while True:
    from_function = find_hand()
    img_original = from_function[0]
    if(len(from_function[1])!=0):

        hand_region=from_function[1]

        feature[0] = cv.resize(hand_region, (90, 90), interpolation=cv.INTER_CUBIC)
        #feature[0]= cv.GaussianBlur(feature[0], (7, 7), cv.BORDER_DEFAULT)
        prediction = model.predict(feature)
        cv.putText(img_original,str(names[np.argmax(prediction[0])]), (80, 60), cv.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255), 2)

    cv.imshow("image",img_original)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()



