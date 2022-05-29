#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
        hand.py
This file contains the class used to store and manipulate the
landmarks found by mediapipe and is based on the work found in
https://www.youtube.com/watch?v=dux3KcKxtCw&t=274s.
'''
__author__  = 'Ferriol Pey Comas [ferriol73pey@gmail.com]'
__version__ = 'v1.0'
__date__    = '29/05/2022'

import cv2


class Hand():
    '''
			This class is a container for a hand object with all the methods
            that are needed in order to interactuate with it.

			==========
			Parameters
			==========

			   w :  int
			        This specifies the width of the image from which the
                    landmarks have been obtained.

			   h :  int
			        This specifies the height of the image from which the
                    landmarks have been obtained.

               landmarks :  landmark list
 			        This contains the actual landmarks found by mediapipe.

 			   habdedness : classification structure
                    This contains the hand classification (left or right hand).

	'''

    def __init__(self,w,h,landmarks,handedness):
        # Creates an empty array to save the landmarks
        self.landmarks = []
        # Transforms the landmarks into image points
        for landmark in landmarks.landmark:
            self.landmarks.append([int(landmark.x*w), int(landmark.y*h)])
        # Gets if right hand or left hand
        self.handType = handedness.classification[0].index
        # Normalizes the landmarks and saves them
        self.normalized_landmarks=self.normalize_landmarks()

    def normalize_landmarks(self):
        '''

		This method normalizes the landmark points of the class and turns it a
        1D vector.

		=======
		RETURNS
		=======

		   normalizedVector   : list of floats
			   It returns the normalized values.

		'''
        # Relativizes the vector to the hand basis point
        tmpVector = []
        for landmark in self.landmarks:
            tmpVector.append([ landmark[0]-self.landmarks[0][0],
                                landmark[1]-self.landmarks[0][1]])
        # Since the first point is going to be always [0,0] it is removed
        tmpVector=tmpVector[1:]

        # Makes it a 1D vector
        tmpVector = [item for sublist in tmpVector for item in sublist]

        # Gets the maximum absolut value to be used to normalize
        maxValue = max(tmpVector, key=abs)

        # Normalizes it
        tmpVector = [item/maxValue for item in tmpVector]

        # Returns the nromalized vector
        return tmpVector

    def drawHand(self, img):
        '''

		This method draws the hand in the provided image.

        ==========
        Parameters
        ==========

           img  :  image
                The image in which the hand should be drawn.

		'''
        handColor = ( 0, 0, 255) # Red if right hand
        if self.handType ==1:
            handColor = ( 255, 0, 0) # Blue if left hand
        # Draw the hand
        ind = [0,1,5,9,13,17]
        for i in range(len(ind)-1):
            cv2.line(img, self.landmarks[ind[i]], self.landmarks[ind[i+1]], (127, 127 , 127), 3)
        cv2.line(img, self.landmarks[17], self.landmarks[0], (127, 127 , 127), 3)
        # Draw the wrist point as a reference point
        cv2.circle(img, self.landmarks[0], 10, (0, 0 , 0), -10)

        # Draw the fingers
        for i in range(5):
            for j in range(1,4):
                p1 = (i*4)+j
                p2 = (i*4)+j+1
                cv2.line(img, self.landmarks[p1], self.landmarks[p2], ( 127, 127, 127), 3)

        # Draw the joints
        for landmark in self.landmarks:
            cv2.circle(img, landmark, 4, handColor, 3)
