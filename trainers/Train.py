import sys, os
parent_dir = os.getcwd()
sys.path.append("/home/ip/IPNeuronaleNetze")
sys.path.append("/home/ip/IPNeuronaleNetze/trainers")
import tensorflow as tf

from models.optimizer.LR_SGD import LR_SGD
from models.OurModel import OurModel
from models.dataloaders.DataLoader import DataLoader
from Trainer import Trainer

#filepathGender = "/home/ip/IPNeuronaleNetze/data/gender.tfrecords"
#filepathGendervalidation = "/home/ip/IPNeuronaleNetze/data/validationgender.tfrecords"

#filepathAge = "/home/ip/IPNeuronaleNetze/data/lap_train.tfrecords"
#filepathAgevalidation = "/home/ip/IPNeuronaleNetze/data/lap_valid.tfrecords"

filepathAge = "/home/ip/IPNeuronaleNetze/data/LAP/Train"
filepathAgevalidation = "/home/ip/IPNeuronaleNetze/data/LAP/Valid"
filepathAgetest = "/home/ip/IPNeuronaleNetze/data/LAP/Test"

imdb_age_test = "/home/ip/IPNeuronaleNetze/data/IMDB/age/test"
imdb_age_train= "/home/ip/IPNeuronaleNetze/data/IMDB/age/train"
imdb_age_val  = "/home/ip/IPNeuronaleNetze/data/IMDB/age/val"

imdb_gender_test = "/home/ip/IPNeuronaleNetze/data/IMDB/gender/test"
imdb_gender_train= "/home/ip/IPNeuronaleNetze/data/IMDB/gender/train"
imdb_gender_val  = "/home/ip/IPNeuronaleNetze/data/IMDB/gender/val"




def main():

    #IMDB Dataset
    #-------------------------------------------------------------------------------------------
    GenderModel = OurModel(1)
    GenderModel = Trainer(GenderModel.model, imdb_gender_train, imdb_gender_val, imdb_gender_test, 1)

    #AgeModel = OurModel(0)
    #AgeModel = Trainer(AgeModel.model, imdb_age_train, imdb_age_val, imdb_age_test,0)



    #LAP Dataset	
    #-------------------------------------------------------------------------------------------
    #AgeModel = OurModel(0)
    #AgeModel  = Trainer(AgeModel.model, filepathAge, filepathAgevalidation, filepathAgetest, 0)

    
     



if __name__ == "__main__":
    main()
