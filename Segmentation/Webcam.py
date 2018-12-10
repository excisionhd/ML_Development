import os
import keras
import cv2
import numpy as np
#from imutils import resize as rz
from skimage.transform import resize
from model import build_model
from keras.models import load_model
from keras.optimizers import Adam
import scipy.misc
from skimage.io import imread, imshow, imread_collection, concatenate_images

vggunet = build_model()

vggunet.load_weights("model-hairsg-vggunet_82dl.h5")
#vggunet.load_weights('model-hairsg-vggunet.h5')

X_test = np.zeros((1,256,256,3), dtype=np.float32)

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    raise IOError("Cannot open webcam.")

while True:
    ret_val, img = cam.read()

    img = cv2.flip(img, 1)

    resized_img = resize(img, (256, 256), mode="constant")

    X_test[0] = resized_img
    mask = vggunet.predict(X_test)

    mask_resized = resize(np.squeeze(mask[0]), (256, 256), mode='constant')

    mask = np.stack((mask_resized,)*3, axis=-1)
    transparency = 0.3
    mask*=transparency

    green = np.ones(resized_img.shape, dtype=np.float)*(0,0,1)

    out = green*mask + resized_img*(1.0-mask)



    cv2.imshow('Output', out)

    if cv2.waitKey(1) == 27: 
        cv2.destroyAllWindows()

