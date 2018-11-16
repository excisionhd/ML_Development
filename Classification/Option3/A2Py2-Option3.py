
# coding: utf-8

# In[1]:


import os
import numpy as np
from numpy import array
from numpy import argmax
import imageio
import matplotlib.pyplot as plt
import pandas as pd
import glob
import keras
import time
from keras import backend as K
from keras.layers.core import Dense
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint 
import matplotlib.pyplot as plt
import cv2
from scipy.misc import imread, imresize
from keras.applications.vgg16 import VGG16
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras import backend as K
from keras.utils import to_categorical
from numpy import genfromtxt

from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


my_data = pd.read_csv('train_data.csv', header=None)


# In[3]:


print(my_data)


# In[4]:


def plotImage(image):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(image)
    axarr[0].grid()
    axarr[0].set_title('Image')


# In[5]:


images = []
pad = ((0,0),)*3 + ((0,2),)


for index, row in my_data.iterrows():
    image = np.reshape(row, (48,48))
    image = imresize(image, (224,224,3))
    image = np.reshape(image, image.shape + (1,))
    images.append(image)
    


# In[6]:


X_train = np.asarray(images)


# In[7]:


targets = pd.read_csv('train_target.csv', header = None)


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X_train, targets, test_size=0.10, random_state=3)


# In[9]:


pad = ((0,0),)*3 + ((0,2),)
X_train = np.pad(X_train, pad, 'constant', constant_values = 0)
X_test = np.pad(X_test, pad, 'constant', constant_values = 0)


# In[10]:


print(X_train.shape)


# In[11]:


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[49]:


#Create the VGG Face model
vgg_face_model = VGGFace(model = 'vgg16', include_top = False, weights='vggface', input_shape=(224,224,3))
    
vgg_face_model.summary()

#Freeze all layers
for layer in vgg_face_model.layers[:-1]:
    layer.trainable = False


# In[56]:


#Add the fully connected layers and classifier
from keras import regularizers
LL = vgg_face_model.get_layer('pool5').output
x = Flatten(name='flatten')(LL)
x = Dense(64,name = 'fc7')(x)
x = Dropout(0.5)(x)
out = Dense(3, activation='softmax',name='classifier')(x)
custom_vgg_face_model = Model(vgg_face_model.input, out)


# In[57]:


#Specify optimizer, loss function, and metrics to track
sgd = SGD(lr=0.0003, decay=1e-5, momentum=0.9)
custom_vgg_face_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


# In[58]:


# fine-tune the model
filepath="vgg_face_gender_weights_improvment-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#tensorboard = TensorBoard(log_dir=".", histogram_freq=2000, write_graph=True, write_images=False)
callback_list = [checkpoint]

#fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
custom_vgg_face_model.fit(x=X_train,y=y_train, batch_size=32, epochs=75, verbose=1, callbacks=callback_list, validation_data=(X_test, y_test))


# In[ ]:


print(X_train[0].shape)


# In[59]:


#load weights if needed
custom_vgg_face_model.load_weights("vgg_face_gender_weights_improvment-06-0.80.hdf5")
print("Loaded model from disk")


# In[147]:


#read file names from sample_submissions
test_data = pd.read_csv('test_data.csv', header=None)


# In[148]:


print(test_data)


# In[149]:


test_data.shape


# In[150]:


test_images = []
pad = ((0,0),)*3 + ((0,2),)


for index, row in test_data.iterrows():
    im = np.reshape(row, (48,48))
    im = imresize(im, (224,224,3))
    im = np.reshape(im, im.shape + (1,))
    test_images.append(im)
    


# In[151]:


test_images2 = np.asarray(test_images)


# In[152]:


pad = ((0,0),)*3 + ((0,2),)
test_images2 = np.pad(test_images2, pad, 'constant', constant_values = 0)


# In[153]:


predictions = custom_vgg_face_model.predict(test_images2)


# In[154]:


arg_predictions = np.argmax(predictions, axis=1)


# In[155]:


emotion = pd.DataFrame(arg_predictions) 


# In[156]:


#read file names from sample_submissions
submission = pd.read_csv('sample-submission.csv')


# In[157]:


print(test_images2.shape)


# In[158]:


result = pd.concat([submission, emotion],axis=1)
result.columns = ['Id', 'C1', 'Category']
result = result.drop(columns=['C1'])


# In[159]:


print(result)


# In[160]:


#Save results to CSV
result.to_csv('answers_op3_vgg_face_2.csv')


# In[114]:


plotImage(test_images2[43])


# In[103]:


print(test_images2[10])


# # Method 2: SVM w/ Landmarks

# In[15]:


import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
import os
import pandas as pd
from sklearn.svm import SVC


# In[16]:


emotions = ["0", "1", "2"] 
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
clf = SVC(kernel='linear', probability=True, tol=1e-3, verbose = True)


# In[17]:


my_data = pd.read_csv('train_data.csv', header=None)
labels = pd.read_csv('train_target.csv', header=None)
labels.rename(columns={labels.columns[0]:'Target'}, inplace=True)

dataframe = pd.concat([my_data, labels],axis=1)


# In[235]:


#Split training and testing 80% and 20% respectively
data_size = len(dataframe.index)
train_size = dataframe - (data_size * 0.2)
count = 0

for index, row in dataframe.iterrows():
    image = np.reshape(row[:-1], (48,48))
    target = str(row['Target'])
    str_index = str(index)

    if count<train_size:
        
        if os.path.isdir('data/train/' + target):
            cv2.imwrite('data/train/' + target + '/' + str_index + '.jpg', image)
        else:
            os.makedirs('data/train/'+ target)
            cv2.imwrite('data/train/' + target + '/' +  str_index + '.jpg', image)
            
        count = count + 1
    
    #For the last 20% of data, copy to validation with same logic as previous.
    else:
        if os.path.isdir('data/validation/' + target):
            cv2.imwrite('data/validation/' + target + '/' + str_index + '.jpg', image)
        else:
            os.makedirs('data/validation/'+ target)
            cv2.imwrite('data/validation/' + target + '/' + str_index + '.jpg', image)
            
        count = count + 1
    


# In[246]:


for index, row in data.iterrows():
    image = np.reshape(row[:-1], (48,48))
    target = str(row['Target'])
    str_index = str(index)
    
    cv2.imwrite('data/all/' + target + "/" + str_index + ".jpg", image)


# In[18]:


def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("data\\all\\%s\\*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    validation = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, validation


# In[19]:


data = {} 
def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,48): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"


# In[20]:


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        print(" Working on %s" %emotion)
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised']) #append image array to training data list
                training_labels.append(emotions.index(emotion))
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels


# In[ ]:


accur_lin = []
for i in range(0,10):
    print("Making sets %s" %i) #Make sets by random sampling 80/20%
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels)
    print("training SVM linear %s" %i) #train SVM
    clf.fit(npar_train, training_labels)
    print("getting accuracies %s" %i) #Use score() function to get accuracy
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print("linear: ", pred_lin)
    accur_lin.append(pred_lin) #Store accuracy in a list
print("Mean value lin svm: %s" %np.mean(accur_lin)) #FGet mean accuracy of the 10 runs

