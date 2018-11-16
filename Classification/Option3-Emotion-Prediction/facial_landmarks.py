#!/usr/bin/env python3
import dlib
import cv2
import numpy as np
import argparse
import imutils
import pandas as pd
import landmarks_helper as lh
from imutils import face_utils

dataset = pd.read_csv('train_data.csv', header=None)

images = []
for index, row in dataset.iterrows():
    image = np.reshape(np.asarray(row), (48, 48))
    images.append(image)
    #cv2.imwrite('images/img_' + str(index) + '.jpg', image)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# detect faces in the grayscale image
image = cv2.imread("images/color_img.jpg")

#image = (image/256).astype('uint8')
image = imutils.resize(image, width=100, height=100)

rects = detector(image, 1)

# show the output image with the face detections + facial landmarks
lh.addLandmarksToImage(image, rects, predictor)

cv2.imshow("Output", image)
cv2.waitKey(0)
