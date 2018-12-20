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
from keras.backend import eval
import numpy as np
from LR_SGD import LR_SGD
from OurModel import OurModel
from keras.legacy import interfaces
from DataLoader import DataLoader
from Trainer import Trainer

filepath = ' '
class modelTest(tf.test.TestCase):
          

    def test_model_layer_length(self):
        with self.test_session():                 
            genderModel = OurModel(1, filepath)
            length=0
            for layer in genderModel.model.layers:
                length = length+1   
            
            trainer = Trainer(genderModel.model, filepath)         
            lengthNew=0
            for layer in trainer.model.layers:
                lengthNew = length+1 
               
            self.assertNotEqual(length,lengthNew)
    
    def test_train_Gender_layers_change(self):
        with self.test_session():                
            genderModel = OurModel(1, filepath)
            weights = []
            for layer in genderModel.model.layers:
                weights.append(layer.weights) 

            trainer = Trainer(genderModel.model, filepath)
            weightsNew = []
            for layer in trainer.model.layers:
                weightsNew.append(layer.weights)            

            self.assertNotEqual(weights,trainer)    

#             modelWeightsAfterTraining = []
#             for layer in model.model.layers:
#                 modelWeightsAfterTraining.append(layer.weights)                 
#             self.assertAllNotEqual(modelWeights , modelWeightsAfterTraining) 
        
#        def test_weights_has_changed(tf.test.TestCase):
#            with self.test_session():
#                model = OurModell(1) 
#
#                  self.assertNotEqual(model, mode_after_train)
        

        
    
    # def test_Age_layersLength(self):
    #     with self.test_session():
    #         AgeModel = OurModel(0)
    #         length=0
    #         for layer in AgeModel.model.layers:
    #             length = length+1       
    #         self.assertEqual(length, 6)

    # def test_Model_layers_allTrainable(self):
    #     with self.test_session():
    #         GenderModel = OurModel(1)
    #         trainable = True
    #         for layer in GenderModel.model.layers:
    #             if layer.trainable != True:
    #                 trainable = False
    #         AgeModel = OurModel(0)
    #         for layer in AgeModel.model.layers:
    #             if layer.trainable != True:
    #                 trainable = False
    #         self.assertEqual(trainable, True)    

#TODO
# class train_modelTest(tf.test.TestCase):
#     def test_trainable_weights(self):
#         with self.test_session():
#             model = OurModel(1)  
 
#             modelWeights = []
#             for layer in model.model.layers:
#                 modelWeights.append(layer.weights) 

#             trainer = Trainer(model.model, 'C:\\Users\\Home\\Desktop\\IP\\gender.tfrecords')
#             trainer.training()

#             modelWeightsAfterTraining = []
#             for layer in model.model.layers:
#                 modelWeightsAfterTraining.append(layer.weights)                 
#             self.assertAllEqual(modelWeights , modelWeightsAfterTraining) 
            
#        def test_weights_has_changed(tf.test.TestCase):
#            with self.test_session():
#                model = OurModell(1) 
#
#                  self.assertNotEqual(model, mode_after_train)

if __name__ == '__main__':
    tf.test.main()
  
   
    