import cv2 as cv
import os
import csv

import tensorflow as tf
import numpy as np
from datetime import datetime, timedelta

print("-> Starting read of metadata file")
dataset = input("train, valid, test?")
type_data = dataset;
csv_data = "gt_avg_" + type_data + ".csv"
age = []
full_path =[]

with open(csv_data, "r") as f:
    reader = csv.reader(f, delimiter="\\")
    for i, line in enumerate(reader):
        if i != 0:
            age_string = line[0].split(',')[4]
            age_float = float(age_string)
            age.append(age_float)
            path = line[0].split(',')[0]
            full_path.append(path)


print("-> metadata file read complete, creating data arrays")
print(age[0])
type(age[0])
I = {}
for idx, val in enumerate(full_path):
    try:
        I[val] = {"age": age[idx]}
    except:
        I[val] = {"age": -1}

    I[val]["img"] = cv.imread(type_data + "/" + val)
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
print("-> " + type_data + " set ready")

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
filename = 'lap-' + type_data + ' set for tfrecords'


writer = tf.python_io.TFRecordWriter(train_filename)

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