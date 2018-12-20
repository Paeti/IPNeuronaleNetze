
import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

import sys

def load_images_from_folder(folder):
    images = []
    for subfolder in os.listdir(folder):
        images.append(os.path.join(folder, subfolder))
    return images

# folder where to search for pictures - no non folder/picture files or error
fp = "../data/fg-net_set/FGNET/images"
cur_pad = os.path.normpath(fp)
full_path = load_images_from_folder(cur_pad)

print("-> metadata file read complete, creating data arrays")


age = []
for a in full_path:
    b = a.split('\\')
    b = b[5].split('.')

    c = b[0].split('A')
    d = c[1].strip('a')
    d = d.strip('b')
    age.append(d)

I = {}
for idx, val in enumerate(full_path):
    try:
        I[val] = {"age": age[idx]}
    except:
        I[val] = {"age": -1}

    I[val]["img"] = cv.imread(val)
    try:
        I[val]["img"] = cv.resize(I[val]["img"], (224, 224))
    except:
        del I[val]


print("-> data array creating completed, flushing into training ready dataset")

X = []
Y = []
for k, v in I.items():
    X.append(v["img"])
    Y.append(v["age"])
print("-> " + 'train' + " set ready")

#load images
def load_image(img):
    #img = cv.resize(img, (224, 224), interpolation=cv.INTER_CUBIC)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img


#convert data to features
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


#write data in tfrecords file
filename = 'fg set for tfrecords'


writer = tf.python_io.TFRecordWriter(filename)

for i in range(len(X)):
    img = load_image(X[i])

    '''
    choose the right label you want to train on    
    '''

    label = Y[i]

    label = int(label)

    feature = {'train/label': _int64_feature(label),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
writer.close()

print("-> created and wrote trfrecords file for selected dataset")