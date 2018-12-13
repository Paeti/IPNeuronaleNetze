import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.python.keras.optimizers import SGD
import numpy as np


class buildModel():

    def __init__(classes):

        VGG16model = VGG16(weights="imagenet", include_top=False)
        input = Input(shape=(224, 224, 3), name='imageInput')
        VGG16output = VGG16model(input)
        x = Flatten(name='flatten')(VGG16output)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
        model = Model(inputs=input, outputs=x)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')