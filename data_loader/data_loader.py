import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class DataLoader:
    def __init__(self, data_path):
        feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
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
        images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
        return image, label