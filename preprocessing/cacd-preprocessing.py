import sys
import cv2 as cv
import tensorflow as tf
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

print("-> Starting read of metadata file")

#facelocation for each image in seperate landmark file -> needs to be downloaded
#needs to be added to the following
with open('metadata.csv', 'r') as file:
    lines = [line.rstrip('\n') for line in file]
    # dob, age, path
    age = lines[2]
    full_path = lines[4]

    age = age.lstrip('[').rstrip(']')
    age = age.split(' ')
    age = [float(x) for x in age]
    print(age[0])

    full_path = full_path.split(' ')
    print(full_path[0])

print("-> metadata file read complete, creating data arrays")

I = {}
for idx, val in enumerate(full_path):
    try:
        I[val] = {"age": age[idx]}
    except:
        I[val] = {"age": -1}

#changing path to imdb file !!
    I[val]["img"] = cv.imread("cacd" + "/" + val)
    try:
        I[val]["img"] = cv.resize(I[val]["img"], (224, 224))
    except:
        del I[val]

#print(I["01/nm0000001_rm124825600_1899-5-10_1968.jpg"])

print("-> data array creating completed, flushing into training ready dataset")

X = []
Y = []
for k, v in I.items():
    X.append(v["img"])
    Y.append(v["age"])
print("-> training set ready for splitting")


#size_training = 300000/len(I)
#size_test = 10000/len(I)

size_training = 100000/len(I)

'''
    X_train = images für trainingset
    Y_train = labels für trainingset
    X_val = images für validationset
    Y_val = labels für validationset
    X_test = images für testset
    Y_test = labels für testset
'''
#shuffle default = true
#stratify default = none --> nicht schichtenweise
X_train, X_tmp, Y_train, Y_tmp = train_test_split(
    X, Y, train_size=size_training, random_state=1
)

size_val = 50000/len(X_tmp)

X_val, X_test, Y_val, Y_test = train_test_split(
    X_tmp, Y_tmp, train_size=size_val, random_state=1
)

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
train_filename = 'trainset for tfrecords'
validation_filename = 'validationset for tfrecords'
test_filename = 'testset for age.tfrecords'

writer = tf.python_io.TFRecordWriter(train_filename)

for i in range(len(X_train)):
    img = load_image(X_train[i])

    '''
    choose the right label you want to train on    
    '''

    label = Y_train[i]

    label = int(label)

    feature = {'train/label': _int64_feature(label),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
writer.close()

print("-> created and wrote trfrecords file for selected dataset")