import sys, os
sys.path.append("/home/ip/IPNeuronaleNetze")
sys.path.append("/home/ip/IPNeuronaleNetze/trainers")
from models.OurModel import OurModel
from Trainer import Trainer
from data_loader.datasets import Dataset
import tensorflow as tf

# Please enter filepath to the dataset for unittests
# Take in mind to choose a small one. Due to tf.test.TestCase class the single 
# trainingsteps will take longer than normal
filepath = "../data/classification/age/"

class modelTest(tf.test.TestCase):

    # Test whether saved model file will be created after training
    def test_model_get_saved(self):
        with self.test_session():
            model = OurModel(0)
            trainer = Trainer(model.model,filepath + "Train", 
                                filepath + "Valid", filepath + "Test", 
                                identifier = 0, epochs = 1, 
                                save_model = True)
            trainer.train()
            self.assertEqual(True,os.path.exists(trainer.saved_model_path))

    # Test whether model will be saved and loaded properly
    def test_model_get_saved_and_loaded_correctly(self):
        with self.test_session():
            model = OurModel(0)
            trainer = Trainer(model.model,filepath + "Valid",
                    filepath + "Valid", filepath + "Valid",
                    identifier = 0, epochs = 1,
                    save_model = True)
            trainer.train()
            model_load = OurModel(0)
            model_load.load_model(trainer.saved_model_path, 0)
            a = trainer.model.get_weights()
            b = model_load.model.get_weights()
            equal = True
            while len(a) != 0:
                    c = a.pop()
                    d = b.pop()
                    if(c != d).any():
                            equal = False
            self.assertEqual(True,equal)

    # Test whether the weigts are changed after one step of training
    def test_one_training_step(self):
        with self.test_session():
            model = OurModel(0)
            a = model.model.get_weights()
            trainer = Trainer(model.model,filepath + "Train",
                                filepath + "Valid", filepath + "Test",
                                identifier = 0, epochs = 1,
                                save_model = False)
            trainer = trainer.train()
            b = trainer.get_weights()      
            same = True
            while len(a) != 0:
                c = a.pop()
                d = b.pop()
                if (c != d).any():
                    same = False
            self.assertEqual(False, same)

    # Test whether all of the layers of a model are trainable, except the input layers
    def test_model_layers_allTrainable(self):
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

    # Test whether Gendermodel is in expected shape
    def test_Gendermodel_layerLength(self):                                                                                   
        with self.test_session():                                                                                             
            GenderModel = OurModel(1)                                                                                         
            GenderModel.model.summary()                                                                                       
            length=0
            for layer in GenderModel.model.layers:                                                                            
                length = length+1
            self.assertEqual(length,24)  

    # Test whether Gendermodel is in expected shape
    def test_Agemodel_layerLength(self):                                                                                      
        with self.test_session():                                                                                             
            AgeModel = OurModel(0)                                                                               
            length = 0
            for layer in AgeModel.model.layers:                                                                               
                length = length+1
            self.assertEqual(length,24) 

if __name__ == '__main__':
    dataset = Dataset()

    dataset.createFolder("../data/classification")
    dataset.createClassificationFolders("../data/classification/age/Train")
    dataset.createClassificationFolders("../data/classification/age/Valid")
    dataset.createClassificationFolders("../data/classification/age/Test")
    #LAP
    dataset.downloadAndUnzip("http://158.109.8.102/AppaRealAge/appa-real-release.zip", "../data/appa-real-release.zip",     "../data/appa-real-release")
    dataset.readAndPrintDataImagesLAP("../data/appa-real-release", "train", "../data/classification/age/Train")
    dataset.readAndPrintDataImagesLAP("../data/appa-real-release", "valid", "../data/classification/age/Valid")
    dataset.readAndPrintDataImagesLAP("../data/appa-real-release", "test", "../data/classification/age/Test")

    tf.test.main() 