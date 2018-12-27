import sys, os
parent_dir = os.getcwd()
sys.path.append(parent_dir)
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.optimizers import SGD
import numpy as np
from models.optimizer.LR_SGD import LR_SGD
from models.dataloaders.DataLoader import DataLoader
from trainers.Trainer import Trainer




class OurModel:
    def __init__(self, identifier, filepath):
        self.model = self.buildModel(identifier, filepath)              
        self.filepath = filepath

    def buildModel(self, identifier, filepath):   
        dataLoader = DataLoader(filepath)
        image, labels  = dataLoader.create_dataset()
        
        input_layer = Input(tensor=image)
        newModel = VGG16(weights="imagenet", include_top=False)(input_layer) 
       
        # Define the input
        xInput =  newModel  
        # Add the fully-connected layers
        x = Flatten(name='flatten')(xInput)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x) 

        if identifier == 1:
            x = Dense(1, activation='sigmoid', name='predictions')(x)
        else:
            x = Dense(101, activation='softmax', name='predictions')(x)
        
        # Create our own model
        model = Model(inputs=input_layer, outputs=x)

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
                                loss='binary_crossentropy', target_tensors=[labels], metrics=['mae'])
        else:
            model.compile(optimizer= optimizer,
                                loss='categorical_crossentropy', target_tensors=[labels], metrics=['mae'])  
                             
        return model