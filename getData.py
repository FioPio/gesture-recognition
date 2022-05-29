#!/usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp

from time import time as t
from hand import Hand

# Basing in  the class for this


mp_hands = mp.solutions.hands

DATA_FILE = 'training_data/data'
timeWindowSize = 30

FPS = 20.0
MINFREQ = (1/FPS)* 0.98 # Getting some margin just in case

# Frames to store
FRAMESSTORE = FPS


def main():
    # For webcam input:
    cap = cv2.VideoCapture(0)
    timeWindow = []
    num = -1
    remainigFrames = -1
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
                    #print(f'[INFO] {avFPS:.2f}fps, len {len(timeWindow)}')

                # Showing the image
                cv2.imshow('MediaPipe Hands', image)

                #wait and get the key pressed
                key = cv2.waitKey(5)
                if remainigFrames == -1:
                    if key & 0xFF == 27: # If ESC is pressed, exit
                        break
                    elif key >= ord('0') and key<=ord('9'): # If a number is pressed
                        num = int(chr(key))                 # Store the number
                        print (f'[INFO] Adding gesture to {num}')
                        remainigFrames=FRAMESSTORE

                else:
                    if len(handsList)>0:
                        with open(DATA_FILE+str(num)+'.txt', 'a') as f:
                            f.write(f'{handsList[-1].normalized_landmarks}')
                            f.close()
                        remainigFrames = remainigFrames-1
                        if remainigFrames==0:
                            with open(DATA_FILE+str(num)+'.txt', 'a') as f:
                                f.write('\n')
                                f.close()
                            remainigFrames = -1
                            num = -1
                            print('[INFO] Stoped the recording')

    cap.release()



if __name__ == '__main__':
    main()
