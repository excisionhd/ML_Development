import os
import keras
import cv2
import numpy as np
from skimage.transform import resize
from model import build_model
from keras.models import load_model
from keras.optimizers import Adam
import scipy.misc
from skimage.io import imread, imshow, imread_collection, concatenate_images

#Call build_model function from model.py
vggunet = build_model()

#Load pre-trained weights from assignment 3
vggunet.load_weights("weights.h5")

#Create a zeroes array for storing prediction.
X_test = np.zeros((1,256,256,3), dtype=np.float32)

#Get OpenCV Webcam stream
cam = cv2.VideoCapture(0)

#Check if video exists
if not cam.isOpened():
    raise IOError("Cannot open webcam.")

while True:
    #Read images from webcam
    ret_val, img = cam.read()

    #Flip the image
    img = cv2.flip(img, 1)

    #Resize the image to fit the model
    resized_img = resize(img, (256, 256), mode="constant")

    #Store image into array to prepare for prediction
    X_test[0] = resized_img

    #Generate the mask with the model
    mask = vggunet.predict(X_test)

    #Resize the mask to 256x256
    mask_resized = resize(np.squeeze(mask[0]), (256, 256), mode='constant')

    #Convert grayscale 2D image into RGB 3D Image (A necessary step to apply alpha blending)
    mask = np.stack((mask_resized,)*3, axis=-1)

    #Specify transparency level (how transparent/see-through the mask will be)
    transparency = 0.3
    mask*=transparency

    #Specify the color change of the mask (B, G, R) (To get more precise colors, divide pixel values by 255)
    green = np.ones(resized_img.shape, dtype=np.float)*(1,0,0.5)

    #Generate new image with mask overlay/blend
    out = green*mask + resized_img*(1.0-mask)

    #Show generated image
    cv2.imshow('Output', out)

    if cv2.waitKey(1) == 27: 
        cv2.destroyAllWindows()
