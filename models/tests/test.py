import sys, os
parent_dir = os.getcwd()
sys.path.append(parent_dir)
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.backend import eval
import numpy as np
from LR_SGD import LR_SGD
from keras.legacy import interfaces
class OurModel:
    def __init__(self, identifier):
        self.model = self.buildModel(identifier)

    def buildModel(self, identifier):
        input_layer = Input(shape=(224,224,3))
        newModel = VGG16(weights="imagenet", include_top=False)(input_layer) #input_shape = (224, 224, 3)
       
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
        model1 = Model(inputs=input_layer, outputs=x)

        # Setting the Learning rate multipliers
        LR_mult_dict = {}
        LR_mult_dict['flatten'] = 100
        LR_mult_dict['fc1'] = 100
        LR_mult_dict['fc2'] = 100   
        LR_mult_dict['predictions'] = 100 

        # Setting optimizer for model        
        optimizer = LR_SGD(lr=0.0001, momentum=0.9, decay=0.0005, nesterov=True, multipliers = LR_mult_dict)
        #optimizer = SGD(lr=0.0001, decay=0.0005,
        #                           momentum=0.9, nesterov=True)  

        # Optimize VGG16 for gender- and agemodel
        if identifier == 1:
            model1.compile(optimizer= optimizer,
                                loss='binary_crossentropy')
        else:
            model1.compile(optimizer= optimizer,
                                loss='categorical_crossentropy')  
        return model1

class build_modelTest(tf.test.TestCase):
    # def test_Age_differently_than_VGG16(self):
    #     with self.test_session():
    #         VGG16model = VGG16(weights="imagenet", include_top=False)
    #         AgeModel = OurModel(0)            
    #         self.assertNotEqual(VGG16model, AgeModel.model)
    
    # def test_Gender_differently_than_VGG16(self):
    #     with self.test_session():
    #         VGG16model = VGG16(weights="imagenet", include_top=False)
    #         GenderModel = OurModell(1)
    #         self.assertNotEqual(VGG16model, GenderModel.model)

    def test_Gender_layersLength(self):
        with self.test_session():
            GenderModel = OurModel(1)
            length=0
            for layer in GenderModel.model.layers:
                length = length+1       
            self.assertEqual(length, 6)
    
    def test_Age_layerLength(self):
        with self.test_session():
            AgeModel = OurModel(0)
            length=0
            for layer in AgeModel.model.layers:
                length = length+1       
            self.assertEqual(length, 6)

    def test_Model_layers_allTrainable(self):
        with self.test_session():
            GenderModel = OurModel(1)
            trainable = True
            for layer in GenderModel.model.layers:
                if layer.trainable != True:
                    trainable = False
            AgeModel = OurModel(0)
            for layer in AgeModel.model.layers:
                if layer.trainable != True:
                    trainable = False
            self.assertEqual(trainable, True)    

#TODO
# class train_modelTest(tf.test.TestCase):
#     def test_trainable_weights(self):
#         with self.test_session():
#             model = OurModell(1)  
#  
#             modelWeights = []
#             for layer in model.model.layers:
#                 modelWeights.append(layer.weights) 

#             trainer = Trainer(model, filepath, save_model_path....)
#             trainer.training()

#             modelWeightsAfterTraining = []
#             for layer in model.model.layers:
#                 modelWeightsAfterTraining.append(layer.weights)                 
#             self.assertAllNotEqual(modelWeights , modelWeightsAfterTraining) 
            
#        def test_weights_has_changed(tf.test.TestCase):
#            with self.test_session():
#                model = OurModell(1) 
#
#                  self.assertNotEqual(model, mode_after_train)

if __name__ == '__main__':
    tf.test.main()
  
   
    