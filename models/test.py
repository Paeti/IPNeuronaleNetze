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
from optimizer.LR_SGD import LR_SGD
from OurModel import OurModel
from keras.legacy import interfaces
from DataLoader import DataLoader
from Trainer import Trainer

filepath = 'C:\\Users\\ckrem\\Desktop\\IP\\age.tfrecords '
class modelTest(tf.test.TestCase):
    def test_one_training_step_AllEqual(self):
        with self.test_session():
            model = OurModel(1, filepath)          
            trainer = Trainer(model.model, filepath, 1)             
            a = model.model.get_weights()
            b = trainer.model.get_weights() 
            self.assertAllEqual(a,b)

    def test_one_training_step(self):
        with self.test_session():            
            model = OurModel(1, filepath)          
            trainer = Trainer(model.model, filepath, 1)             
            a = model.model.get_weights()
            b = trainer.model.get_weights()   
            same = True         
            while len(a) != 0:
                c = a.pop()
                d = b.pop()
                if (c != d).any():
                    same = False
            self.assertEqual(False, same)

    def test_Model_layers_allTrainable(self):
        with self.test_session():
            GenderModel = OurModel(1, filepath)
            trainable = True
            for layer in GenderModel.model.layers:                
                if layer.trainable != True and layer.name != 'input_1':
                    trainable = False
            AgeModel = OurModel(0, filepath)
            for layer in AgeModel.model.layers:
                if layer.trainable != True  and layer.name != 'input_3':
                    trainable = False
            self.assertEqual(trainable, True)         

    def test_AgeModel_layerLength(self):
        with self.test_session():
            AgeModel = OurModel(1, filepath)                     
            length=0            
            for layer in AgeModel.model.layers:
                length = length+1 
            self.assertEqual(length,6)

    def test_GenderModel_layerLength(self):
        with self.test_session():
            GenderModel = OurModel(0, filepath)
            length = 0
            for layer in GenderModel.model.layers:
                length = length+1 
            self.assertEqual(length,6)

    def test_loss_AgeModel(self):
        with self.test_session():
            AgeModel = OurModel(1, filepath) 
            AgeModel = Trainer(AgeModel.model, filepath, 1)            
            self.assertNotEqual(AgeModel.model.loss,0)
    
    def test_loss_GenderModel(self):
        with self.test_session():
            GenderModel = OurModel(0, filepath) 
            GenderModel = Trainer(GenderModel.model, filepath, 0)            
            self.assertNotEqual(GenderModel.model.loss,0)
       



if __name__ == '__main__':
    tf.test.main()
  
   
    