from keras.models import Model, load_model
from keras.layers import Input, Dropout, Activation, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Dropout




def build_model():
    #Import Pretained VGG16
    base_pretrained_model = VGG16(input_shape =  (256,256,3), include_top = False, weights = 'imagenet')
    base_pretrained_model.trainable = False
    #base_pretrained_model.summary()

    #Add CONV2D Layer with Batch Normalization
    x = base_pretrained_model.layers[-1].output
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='single_conv')(x)
    x = (BatchNormalization())(x)

    vggunet_custom = Model(base_pretrained_model.input, x)
    #Freeze first 5 convolution blocks
    for layer in vggunet_custom.layers[:-3]:
        layer.trainable = False

    #Prepare skip connections
    f5 = vggunet_custom.get_layer("block5_conv1").output
    f4 = vggunet_custom.get_layer("block4_conv1").output
    f3 = vggunet_custom.get_layer("block3_conv1").output
    f2 = vggunet_custom.get_layer("block2_conv1").output
    f1 = vggunet_custom.get_layer("block1_conv1").output

    #Upsample followed by CONV2D Transpose, Skip Connection, and finish the CONV block with Batch Normalization
    o = UpSampling2D(size = (2,2))(x)
    up6 = Conv2DTranspose(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(o)
    merge6 = concatenate([f5, up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    o = (BatchNormalization())(conv6)

    o = UpSampling2D(size = (2,2))(o)
    up6 = Conv2DTranspose(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(o)
    merge6 = concatenate([f4, up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    o = (BatchNormalization())(conv6)

    o = UpSampling2D(size = (2,2))(o)
    up6 = Conv2DTranspose(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(o)
    merge6 = concatenate([f3, up6], axis = 3)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    o = (BatchNormalization())(conv6)

    o = UpSampling2D(size = (2,2))(o)
    up6 = Conv2DTranspose(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(o)
    merge6 = concatenate([f2, up6], axis = 3)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    o = (BatchNormalization())(conv6)

    o = UpSampling2D(size = (2,2))(o)
    up6 = Conv2DTranspose(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(o)
    merge6 = concatenate([f1, up6], axis = 3)
    conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    o = (BatchNormalization())(conv6)

    o = Conv2D(1, (1,1), activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(o)

    vggunet_custom = Model(vggunet_custom.input, o)
    return vggunet_custom

