from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers


class Model:
    def __init__(self, config, classes):
        self.build_model((224, 224, 3), classes)

    def build_model(input_shape, classes):
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape))
        model.add(Convolution2D(64, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, LeakyReLU(alpha=0.3)))
        model.add(Dropout(0.5))
        model.add(Dense(4096, LeakyReLU(alpha=0.3)))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        return model
