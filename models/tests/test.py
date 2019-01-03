import sys, os
parent_dir = os.getcwd()
sys.path.append("/Users/ronnyaretz/IPNeuronaleNetze")
sys.path.append("/Users/ronnyaretz/IPNeuronaleNetze/trainers")
import tensorflow as tf
import numpy as np
from models.optimizer.LR_SGD import LR_SGD
from models.OurModel import OurModel
from models.dataloaders.DataLoader import DataLoader
from Trainer import Trainer

filepathGender = "IPNeuronaleNetze/data/gender.tfrecords"
filepathGendervalidation = "IPNeuronaleNetze/data/validationgender.tfrecords"

filepathAge = "IPNeuronaleNetze/data/age.tfrecords"
filepathAgevalidation = "IPNeuronaleNetze/data/validationage.tfrecords"

class modelTest(tf.test.TestCase):

    # Test whether weights files get created after training
    def test_weights_get_saved(self):
        with self.test_session():
            stored = True
            GenderModelWeights = "IPNeuronaleNetze/models/weights/GenderModel_weights.h5"
            GenderModel = OurModel(1, filepathGender)
            GenderModel = Trainer(GenderModel.model, filepathGender, 1)
            with open(GenderModelWeights) as weightsfile:
                first = weightsfile.read(1)
                if not first:
                    stored = False            

            AgeModelWeights = "IPNeuronaleNetze/models/weights/AgeModel_weights.h5"
            AgeModel = OurModel(0, filepathAge)
            AgeModel = Trainer(AgeModel.model, filepathAge, 0)
            with open(AgeModelWeights) as weightsfile:
                first = weightsfile.read(1)
                if not first:
                    stored = False
                
            self.assertEqual(True, stored)

    # Test whether the Model weights are changed after training with one step
    def test_one_training_step(self):
        with self.test_session():            
            model = OurModel(1, filepathGender) 
            a = model.model.get_weights()         
            trainer = Trainer(model.model,filepathGender , 1) 
            b = trainer.model.get_weights()   
            same = True         
            while len(a) != 0:
                c = a.pop()
                d = b.pop()
                if (c != d).any():
                    same = False
            self.assertEqual(False, same)

    # Test whether all of the layers of a model are trainable, except the input layers
    def test_Model_layers_allTrainable(self):
        with self.test_session():
            GenderModel = OurModel(1, filepathGender)
            trainable = True
            for layer in GenderModel.model.layers:                
                if layer.trainable != True and layer.name != 'input_1':
                    trainable = False
            AgeModel = OurModel(0, filepathAge)
            for layer in AgeModel.model.layers:
                if layer.trainable != True  and layer.name != 'input_3':
                    trainable = False
            self.assertEqual(trainable, True)         

    # Test wether the models layer length are in expected manner
    def test_GenderModel_layerLength(self):
        with self.test_session():
            GenderModel = OurModel(1, filepathGender)                     
            length=0            
            for layer in GenderModel.model.layers:
                length = length+1 
            self.assertEqual(length,23)

    # Test wether the models layer length are in expected manner
    def test_AgeModel_layerLength(self):
        with self.test_session():
            AgeModel = OurModel(0, filepathAge)
            length = 0
            for layer in AgeModel.model.layers:
                length = length+1 
            self.assertEqual(length,23)

    # Test wether the loss of the models is not equal 0
    def test_loss_AgeModel(self):
        with self.test_session():
            GenderModel = OurModel(1, filepathGender) 
            GenderModel = Trainer(GenderModel.model, filepathGender, 1)            
            self.assertNotEqual(GenderModel.model.loss,0)
    
    # Test wether the loss of the models is not equal 0
    def test_loss_GenderModel(self):
        with self.test_session():
            AgeModel = OurModel(0, filepathAge) 
            AgeModel = Trainer(AgeModel.model, filepathAge, 0)            
            self.assertNotEqual(AgeModel.model.loss,0)


if __name__ == '__main__':
    tf.test.main() 