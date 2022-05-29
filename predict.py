#!/usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
from time import time as t
from hand import Hand

import tensorflow as tf
import numpy as np
from pynput.keyboard import Key, Controller

keyboard = Controller()

mp_hands = mp.solutions.hands

MODEL = 'models/modelv3.hdf5'
timeWindowSize = 30

THRESHOLD = 0.80

MAXPREDICTIONSWINDOW = 10


def main():
    # Loading the model
    model = tf.keras.models.load_model(MODEL)
    # For webcam input:
    cap = cv2.VideoCapture(0)
    timeWindow = []
    scrollingMenu = False
    lastPredictions = []
    gestureActive = True
    with mp_hands.Hands( model_complexity=0,
                         min_detection_confidence=0.5,
                         min_tracking_confidence=0.5) as hands: #max_num_hands = 1
        while cap.isOpened():
            # Saves the times to get the framerate
            timeWindow.append(t())
            # If it is bigger than the accepted size just cut the extra part
            if len(timeWindow)>timeWindowSize:
                timeWindow=timeWindow[-timeWindowSize:]
            success, image = cap.read()
            if not success:
                print('[INFO] Ignoring empty camera frame.')
                # If loading a video, use 'break' instead of 'continue'.
                continue
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # list of found hands
            handsList = []
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    image_width, image_height = image.shape[1], image.shape[0]

                    # Creates a custom object hand and append it to hands
                    handsList.append(Hand(image_width, image_height,hand_landmarks, handedness))
                    # Shows the found hand
                    handsList[-1].drawHand(image)
                    if handsList[-1].handType ==0: # Right hand
                        # Predicts
                        prediction = model(np.array([handsList[-1].normalized_landmarks]))[0]
                        index_max = int(tf.argmax(prediction))
                        # Saves the prediction as predicted label, probability
                        prediction = [index_max,float(prediction[index_max])]
                        #print(prediction)
                        if prediction[1]>THRESHOLD:
                            lastPredictions.append(prediction[0])
                            if len(lastPredictions)>MAXPREDICTIONSWINDOW :
                                lastPredictions=lastPredictions[-MAXPREDICTIONSWINDOW : ]
                                # If all elements are equal
                                if lastPredictions.count(lastPredictions[0]) == len(lastPredictions):
                                    if gestureActive:
                                        print(f'[INFO] identifyed {prediction[0]}')
                                        if lastPredictions[0] == 1:
                                            if not scrollingMenu:
                                                keyboard.press(Key.alt)
                                                keyboard.press(Key.tab)
                                                keyboard.release(Key.tab)
                                                scrollingMenu=True
                                            else:
                                                keyboard.release(Key.alt)
                                                scrollingMenu=False
                                        elif lastPredictions[0]==2 or lastPredictions[0]==3:
                                            keyboard.press(Key.left)
                                            keyboard.release(Key.left)
                                        elif lastPredictions[0]==4:
                                            keyboard.press(Key.right)
                                            keyboard.release(Key.right)
                                        elif lastPredictions[0]==5:
                                            if scrollingMenu:
                                                keyboard.release(Key.alt)
                                            exit()
                                    # Switches when the robot is working
                                    if lastPredictions[0] == 7:
                                        gestureActive=not gestureActive
                                        print(f'[INFO] Switching gesture recognition to {gestureActive}')

                                    lastPredictions = []

            # Flip the image horizontally for a selfie-view display.
            image =  cv2.flip(image, 1)
            # Compute and write fps
            l = len(timeWindow)-1
            if l>0:
                avFPS =0
                for i in range(l,0,-1):
                    avFPS += (timeWindow[i]-timeWindow[i-1])
                avFPS = l/avFPS
                # Write FPS
                text = f'{avFPS:.2f}fps'
                cv2.putText(image, text, (15, 25), cv2.FONT_HERSHEY_TRIPLEX, 1, ( 0, 0, 0), 1)

            # Showing the image
            cv2.imshow('MediaPipe Hands', image)

            #wait and get the key pressed
            key = cv2.waitKey(5)
            if key & 0xFF == 27: # If ESC is pressed, exit
                break

    cap.release()


if __name__ == '__main__':
    main()
