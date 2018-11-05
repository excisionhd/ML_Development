#!/usr/bin/env python3
import dlib
import cv2
import numpy as np
import argparse
import imutils
import pandas as pd
from imutils import face_utils

dataset = pd.read_csv('train_data.csv', header=None)

images = []
for index, row in dataset.iterrows():
    image = np.reshape(np.asarray(row), (48, 48))
    images.append(image)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# detect faces in the grayscale image
image = cv2.imread("color_img.jpg")
cv2.imshow('output' , image)
cv2.waitKey(0)

#image = (image/256).astype('uint8')
image = imutils.resize(image, width=100, height=100)

rects = detector(image, 1)



#cv2.imwrite('color_img.jpg', images[0])


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x,y,w,h)

def shape_to_np(shape, dtype="int"):
    coords = np.zeores((68.2), dtype=dtype)

    for i in range(0,68):
        coords[i] = (shape.part(i).x, shape.part(i).y)


    return coords

def addLandmarksToImage(image, rects):
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(image, rect)
		shape = face_utils.shape_to_np(shape)
	
		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	
		# show the face number
		cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

 
# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)
