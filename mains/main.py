import tensorflow as tf
import sys, os
parent_dir = os.getcwd()
sys.path.append(parent_dir)

from models.model import OurModel
from data_loaders.data_loader import DataLoader
from trainers.trainer import Trainer
from tensorflow.python.keras.backend import eval

def trainModel(model, data):
    train_loss_results = []
    train_accuracy_results = []
    epoch_loss_avg = tf.metrics.Mean()
    epoch_accuracy = tf.metrics.Accuracy()

    # ---- TODO ---


def main():
    # create tensorflow session
    sess = tf.Session()
    
    # create an instance of the model you want
    genderModel = OurModel(1)
    ageModel = OurModel(0)
    genderModel.model.summary()
    ageModel.model.summary()
   
    genderModel.model.fit()
    # create trainer and pass all the previous components to it
    trainModel(sess, genderModel)
    trainModel(sess, ageModel)


if __name__ == '__main__':
    main()
