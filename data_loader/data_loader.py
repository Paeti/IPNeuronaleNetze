import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class DataLoader:
    def __init__(self, filepath):
        dataset = tf.data.TFRecordDataset(filepath)
        dataset = dataset.map(_parse_function, num_parallel_calls=8)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(30)
        dataset = dataset.batch(10)
        iterator = dataset.make_one_shot_iterator()
        image = tf.reshape(image, [224, 224, 3])
        image, label = iterator.get_next()
        label = tf.one_hot(label, 2)
        return image, label

    def _parse_function(proto):
        # define your tfrecord again. Remember that you saved your image as a string.
        keys_to_features = {'train/image': tf.FixedLenFeature([], tf.string),
                            "train/label": tf.FixedLenFeature([], tf.int64)}
        # Load one example
        parsed_features = tf.parse_single_example(proto, keys_to_features)
        # Turn your saved image string into an array
        parsed_features['train/image'] = tf.decode_raw(
            parsed_features['train/image'], tf.uint8)
        return parsed_features['train/image'], parsed_features["train/label"]