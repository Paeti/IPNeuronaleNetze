import sys, os
parent_dir = os.getcwd()
sys.path.append("/Users/ronnyaretz/IPNeuronaleNetze")
sys.path.append("/Users/ronnyaretz/IPNeuronaleNetze/trainers")
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.python.keras.optimizers import SGD
import numpy as np
from models.optimizer.LR_SGD import LR_SGD
from models.dataloaders.DataLoader import DataLoader
from Trainer import Trainer


class OurModel:
    def __init__(self, identifier, filepath):
        self.model = self.buildModel(identifier, filepath)              
        self.filepath = filepath

    def buildModel(self, identifier, filepath):   
        # LOAD VGG16
        base_model = VGG16(weights='imagenet', include_top=False)
        fyipg = base_model.output

        # add a global spatial average pooling layer
        fyipg = GlobalAveragePooling2D()(fyipg)
        
        # let's add a fully-connected layer
        fyipg = Dense(4096, activation='relu', name='AdditianlLayer1')(fyipg)
        fyipg = Dense(4096, activation='relu', name='AdditianlLayer2')(fyipg)
        if identifier == 1:
            fyipg = Dense(1, activation='sigmoid', name='predictions')(fyipg)
        else:
            fyipg = Dense(101, activation='softmax', name='predictions')(fyipg)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=fyipg)

        # Setting the Learning rate multipliers
        LR_mult_dict = {}
        LR_mult_dict['flatten'] = 100
        LR_mult_dict['fc1'] = 100
        LR_mult_dict['fc2'] = 100   
        LR_mult_dict['predictions'] = 100 

        # Setting optimizer for model        
        optimizer = LR_SGD(lr=0.0001, momentum=0.9, decay=0.0005, nesterov=True, multipliers = LR_mult_dict)

        # Optimize model for gender- and agemodel
        if identifier == 1:
            model.compile(optimizer= optimizer,
                                loss='binary_crossentropy', metrics=['mae'])
        else:
            model.compile(optimizer= optimizer,
                                loss='categorical_crossentropy', metrics=['mae'])  
                             
        return model