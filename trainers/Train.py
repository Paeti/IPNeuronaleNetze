import sys, os
parent_dir = os.getcwd()
sys.path.append("IPNeuronaleNetze")
sys.path.append("IPNeuronaleNetze/trainers")
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

def main():

    GenderModel = OurModel(1, filepathGender)
    GenderModel = Trainer(GenderModel.model, filepathGendervalidation, filepathGender, 1)

    AgeModel = OurModel(0, filepathAge)
    AgeModel = Trainer(AgeModel.model, filepathAgevalidation, filepathAge, 0)


if __name__ == "__main__":
    main()