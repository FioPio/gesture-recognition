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

MODEL = 'models/modelv2.hdf5'
timeWindowSize = 30

THRESHOLD = 0.80

FPS = 20.0
MINFREQ = (1/FPS)* 0.98 # Getting some margin just in case

# Frames to store
FRAMESSTORE = FPS
# Store 20 points * 2 coordinates (x ,y) * FRAMESSTORE frames
GESTURE_WINDOW_MAX_SIZE = int(20*2*FRAMESSTORE)

def main():
    # Loading the model
    model = tf.keras.models.load_model(MODEL)
    # For webcam input:
    cap = cv2.VideoCapture(0)
    timeWindow = []
    gestureWindow=[]
    scrollingMenu = False
    with mp_hands.Hands( model_complexity=0,
                         min_detection_confidence=0.5,
                         min_tracking_confidence=0.5) as hands: #max_num_hands = 1
        while cap.isOpened():
            # Trying to get a more stable framerate
            if (len (timeWindow)<1) or (t()-timeWindow[-1]>=MINFREQ ):
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
                        # Adds it to the window
                        gestureWindow.extend(handsList[-1].normalized_landmarks)

                        if len(gestureWindow)> GESTURE_WINDOW_MAX_SIZE:
                            gestureWindow=gestureWindow[-GESTURE_WINDOW_MAX_SIZE:]
                            prediction = model(np.array([gestureWindow]))[0]
                            index_max = int(tf.argmax(prediction))
                            # Saves the prediction as predicted label, probability
                            prediction = [index_max,float(prediction[index_max])]
                            #print(prediction)
                            if prediction[1]>THRESHOLD:
                                print(f'[INFO] identifyed {prediction[0]}')
                                # Removes the window to avoid repetitions
                                gestureWindow=[]
                                if prediction[0] == 1:
                                    if not scrollingMenu:
                                        keyboard.press(Key.alt)
                                        keyboard.press(Key.tab)
                                        keyboard.release(Key.tab)
                                        scrollingMenu=True
                                    else:
                                        keyboard.release(Key.alt)
                                        scrollingMenu=False
                                elif prediction[0]==2:
                                    keyboard.press(Key.left)
                                    keyboard.release(Key.left)


                # Flip the image horizontally for a selfie-view display.
                image =  cv2.flip(image, 1)
                # Compute and write fps
                l = len(timeWindow)
                if l>1:
                    avFPS =0
                    for i in range(l-1,0,-1):
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
