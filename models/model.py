import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.backend import eval
import numpy as np


class OurModel:
    def __init__(self, identifier):
        self.model = self.buildModel(identifier)

    def buildModel(self, identifier):
        # Setting optimizers for VGG16Model and customModel
        optimizerForVGG16 = SGD(lr=0.0001, decay=0.0005,
                                momentum=0.9, nesterov=True)
        optimizerForCustomModel = SGD(
            lr=0.001, decay=0.0005, momentum=0.9, nesterov=True)
        # Build VGG16 from Caffeemodel with pretrained weights
        VGG16Model = VGG16(weights="imagenet", include_top=False)
        # Optimize VGG16 for gender- and agemodel
        if identifier == 1:
            VGG16Model.compile(optimizer=optimizerForVGG16,
                               loss='binary_crossentropy')
        else:
            VGG16Model.compile(optimizer=optimizerForVGG16,
                               loss='categorical_crossentropy')
        # Define the input
        input = Input(shape=(224, 224, 3), name='imageInput')
        # Use the generated model
        VGG16output = VGG16Model(input)
        # Add the fully-connected layers
        xInput = Input(shape=(7, 7, 512))
        x = Flatten(name='flatten')(xInput)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        # Optimize the appended last layers for gender- and agemodel
        if identifier == 1:
            x = Dense(1, activation='sigmoid', name='predictions')(x)
        else:
            x = Dense(101, activation='softmax', name='predictions')(x)
        customModel = Model(inputs=xInput, outputs=x, name='customModel')
        if identifier == 1:
            customModel.compile(optimizer=optimizerForCustomModel,
                                loss='binary_crossentropy')
        else:
            customModel.compile(optimizer=optimizerForCustomModel,
                                loss='categorical_crossentropy')
        # Create our own model
        outputLayerOfVGG16Model = VGG16Model.get_layer('block5_pool').output
        mergedModels = customModel(outputLayerOfVGG16Model)
        model = Model(inputs=VGG16Model.input, outputs=mergedModels)
        return model
