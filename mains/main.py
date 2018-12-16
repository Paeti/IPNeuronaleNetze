import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.python.keras.optimizers import SGD
import numpy as np



def buildModel(identifier):
    # Build VGG16 from Caffeemodel with pretrained weights
    VGG16model = VGG16(weights="imagenet", include_top=False)
    # Define the input
    input = Input(shape=(224, 224, 3), name='imageInput')
    # Use the generated model
    VGG16output = VGG16model(input)
    # Add the fully-connected layers
    x = Flatten(name='flatten')(VGG16output)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    if identifier == 1:
        x = Dense(1, activation='sigmoid', name='predictions')(x)
    else:
        x = Dense(101, activation='softmax', name='predictions')(x)
    # Create our own model
    model = Model(inputs=input, outputs=x)
    # Optimize the model
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    if identifier == 1:
        model.compile(optimizer=sgd, loss='binary_crossentropy')
    else:
        model.compile(optimizer=sgd, loss='categorical_crossentropy')
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
