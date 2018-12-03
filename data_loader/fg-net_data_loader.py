from random import shuffle
import glob
import cv2
import tensorflow as tf
import numpy as np
import sys

class fg_net_data_loader:

    shuffle_data = True  # shuffle the addresses before saving
    fg_net_data_path = '../data/fg-net_set/FGNET/images/*.jpg'
    # read addresses and labels from the 'train' folder
    addrs = glob.glob(fg_net_data_path)
    labels = [int(addr[len(addr)-6:len(addr)-4])  for addr in addrs]
    # to shuffle data
    if shuffle_data:
        c = list(zip(addrs, labels))
        shuffle(c)
        addrs, labels = zip(*c)

    # Divide the hata into 95% train, 5% validation
    train_addrs = addrs[0:int(0.95 * len(addrs))]
    train_labels = labels[0:int(0.95 * len(labels))]
    val_addrs = addrs[int(0.95 * len(addrs)):]
    val_labels = labels[int(0.95 * len(addrs)):]

    def load_image(addr):
        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
        img = cv2.imread(addr)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        return img

    def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    def _bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    train_filename = 'train.tfrecords'  # address to save the TFRecords file
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(train_filename)
    for i in range(len(train_addrs)):
        # Load the image
        img = load_image(train_addrs[i])
        label = train_labels[i]
        # Create a feature
        feature = {'train/label': _int64_feature(label),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
