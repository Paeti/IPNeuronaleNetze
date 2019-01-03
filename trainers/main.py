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

filepathGender = "/Users/ronnyaretz/IPNeuronaleNetze/data/gender.tfrecords"
filepathAge = "/Users/ronnyaretz/IPNeuronaleNetze/data/age.tfrecords"

def main():

    #GenderModel = OurModel(1, filepathGender)
    #GenderModel = Trainer(GenderModel.model, filepathGender, 1)

    AgeModel = OurModel(0, filepathAge)
    AgeModel = Trainer(AgeModel.model, filepathAge, 0)
    






if __name__ == "__main__":
    main()