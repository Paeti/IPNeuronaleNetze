import sys
sys.path.append("/IPNeuronaleNetze")
sys.path.append("/IPNeuronaleNetze/trainers")
import tensorflow as tf
from models.OurModel import OurModel
from Trainer import Trainer

# Please enter the filepath to your dataset. 
filepath_age = "/IPNeuronaleNetze/data/"
filepath_gender = "/IPNeuronaleNetze/data/"

def main():
    # This main will start the training for both, age as well as gender estimation
    # with the default amount of epochs with the value 312 and an early stopping callback with the patience of 4.    
    # You can change the amount of epochs by typing "epochs = X" into the Trainer constructor.
    # If you want to train a model by loading the weights of an allready trained model, 
    # use the load_model method from the OurModel class.     

    # After the training is done, the models will be saved and evulation files (.csv) will be created.
    # Use the parsers to interpret the .csv files

    # Start training for age estimation              
    AgeModel = OurModel(0)
    AgeModel = Trainer(AgeModel.model, filepath_age + "Train",
                        filepath_age + "Valid", filepath_age + "Test",
                        identifier = 0)
    AgeModel.train()

    # Start training for gender estimation
    GenderModel = OurModel(1)
    GenderModel = Trainer(GenderModel.model, filepath_gender + "Train",
                            filepath_gender + "Valid", filepath_gender + "Test",
                            identifier = 1)
    GenderModel.train()

if __name__ == "__main__":
    main()