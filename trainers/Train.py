import sys, os
parent_dir = os.getcwd()
sys.path.append("/home/ip/IPNeuronaleNetze")
sys.path.append("/home/ip/IPNeuronaleNetze/trainers")
import tensorflow as tf
import numpy as np
from models.optimizer.LR_SGD import LR_SGD
from models.OurModel import OurModel
from models.dataloaders.DataLoader import DataLoader
from Trainer import Trainer

#filepathGender = "/home/ip/IPNeuronaleNetze/data/gender.tfrecords"
#filepathGendervalidation = "/home/ip/IPNeuronaleNetze/data/validationgender.tfrecords"

filepathAge = "/home/ip/IPNeuronaleNetze/data/lap_train.tfrecords"
filepathAgevalidation = "/home/ip/IPNeuronaleNetze/data/lap_valid.tfrecords"

def main():

    #GenderModel = OurModel(1, filepathGender)
    #GenderModel = Trainer(GenderModel.model, filepathGendervalidation, filepathGender, 1)

    AgeModel = OurModel(0, filepathAge)
    AgeModel = Trainer(AgeModel.model, filepathAgevalidation, filepathAge, 0)


if __name__ == "__main__":
    main()
