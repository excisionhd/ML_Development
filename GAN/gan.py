
#!/usr/bin/env python3

#TODO: Create generator
#Research how discriminator works with generator
import os
import scipy
import numpy as np
import random
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Activation, Dense, Reshape, Flatten
from keras.layers.convolutional import Conv2D, UpSampling2D, MaxPooling2D
from keras.layers.advanced_activations import ELU

def discriminator_model():
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(28, 28, 1))
            )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def G(x):
	raise NotImplementedError

def main():
	z = np.random.uniform(-1, 1, 5)
	model = discriminator_model()

	model.summary()

if __name__ == "__main__":
	main()


