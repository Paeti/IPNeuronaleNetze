import os, sys
sys.path.append("/home/ip/IPNeuronaleNetze")
sys.path.append("/home/ip/IPNeuronaleNetze/trainers")
from data_loader.datasets import Dataset
import tensorflow as tf
from models.OurModel import OurModel
from trainers.Trainer import Trainer

# Please enter the filepath to your dataset.
filepath = "../data/classification/age/"



def main():
    # This main will first load the datasets then start the training for both, age as well as gender estimation
    # with the default amount of epochs which is the value 312 and an early stopping callback with the patience of 4.
    # You can change the amount of epochs by typing "epochs = X" into the Trainer constructor.
    # If you want to train a model by loading the weights of an allready trained model,
    # use the load_model method from the OurModel class.

    # After the training is done, the models will be saved and evulation files (.csv) will be created.
    # Use the parsers to interpret the .csv files


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

    # Start training for age estimation
    AgeModel = OurModel(0)
    AgeModel = Trainer(AgeModel.model, filepath + "Train",
                        filepath + "Valid", filepath + "Test",
                        identifier = 0)
    AgeModel.train()

   

if __name__ == "__main__":
    main()