import tensorflow as tf
import sys, os
parent_dir = os.getcwd()
sys.path.append(parent_dir)

from models.model import OurModel
from data_loaders.data_loader import DataLoader
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
    # create tensorboard logger
    #logger = Logger(sess, config)
        # create your data generator

    genderData = DataLoader("/Users/max/Desktop/trainsetForGender.tfrecords")
    ageData = DataLoader("/Users/max/Desktop/trainsetForAge.tfrecords")
    print(genderData.images)
    print(genderData.labels)
    # create trainer and pass all the previous components to it
    """ trainModel(sess, genderModel, genderDataImage, genderDataLabel)
    trainModel(sess, ageModel, ageDataImage, ageDataLabel) """


if __name__ == '__main__':
    main()
