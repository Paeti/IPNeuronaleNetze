import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.backend import eval
from tensorflow.python.keras.backend import eval
import numpy as np


def buildModel(identifier):
    # Setting optimizers for VGG16Model and customModel
    optimizerForVGG16 = SGD(lr=0.0001, decay=0.0005,
                            momentum=0.9, nesterov=True)
    optimizerForCustomModel = SGD(
        lr=0.001, decay=0.0005, momentum=0.9, nesterov=True)
    # Build VGG16 from Caffeemodel with pretrained weights
    VGG16Model = VGG16(weights="imagenet", include_top=False)
    # 
    if identifier == 1:
        VGG16Model.compile(optimizer=optimizerForVGG16,
                           loss='binary_crossentropy')
    else:
        VGG16Model.compile(optimizer=optimizerForVGG16,
                           loss='categorical_crossentropy')
    print(eval(VGG16Model.optimizer.lr))    
    # Define the input
    input = Input(shape=(224, 224, 3), name='imageInput')
    # Use the generated model
    VGG16output = VGG16Model(input)
    # Add the fully-connected layers
    xInput = Input(shape=(7, 7, 512))
    x = Flatten(name='flatten')(xInput)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    if identifier == 1:
        x = Dense(1, activation='sigmoid', name='predictions')(x)
    else:
        x = Dense(101, activation='softmax', name='predictions')(x)
    customModel = Model(inputs = xInput, outputs = x, name='customModel')
    if identifier == 1:
       customModel.compile(optimizer=optimizerForCustomModel,
                           loss='binary_crossentropy')
    else:
        customModel.compile(optimizer=optimizerForCustomModel,
                           loss='categorical_crossentropy')
    # Create our own model
    outputLayerOfVGG16Model = VGG16Model.get_layer('block5_pool').output
    mergedModels = customModel(outputLayerOfVGG16Model)
    model = Model(inputs = VGG16Model.input, outputs= mergedModels)

    return model


def generateData(datapath):
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([datapath], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/image'], tf.float32)
    # Cast label data into int32
    label = tf.cast(features['train/label'], tf.int32)
    # Reshape image data into the original shape
    image = tf.reshape(image, [224, 224, 3])
    # TODO Any preprocessing here ...
    # Creates batches by randomly shuffling tensors - TODO probably not needed since our data is already pretty random
    images, labels = tf.train.shuffle_batch(
        [image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    # return ??


def trainModel(model, data):
    train_loss_results = []
    train_accuracy_results = []
    epoch_loss_avg = tf.metrics.Mean()
    epoch_accuracy = tf.metrics.Accuracy()

    # ---- TODO ---


def main():
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    genderData = generateData("filpathToGenderTrainset")
    ageData = generateData("filpathToAgeTrainset")
    # create an instance of the model you want
    genderModel = buildModel(1)
    ageModel = buildModel(0)
    # create tensorboard logger
    #logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = trainModel(sess, genderModel, genderData)
    trainer = trainModel(sess, ageModel, ageData)
    # load model if exists
    genderModel.load(sess)
    ageModel.load(sess)
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
