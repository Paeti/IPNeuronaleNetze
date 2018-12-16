
import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

import sys


'''


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
'''

def load_images_from_folder(folder):
    images = []
    for subfolder in os.listdir(folder):
        print(os.path.join(folder, subfolder))
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

#changing path to imdb file !!
    I[val]["img"] = cv.imread("/home/alex/Downloads/CACD_2000_example" + "/" + val)
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



#number of pictures for training set
size_training = 600/len(I)

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

#number of pictures for validation set
size_val = 300/len(X_tmp)

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
train_filename = 'FG trainset for tfrecords'
validation_filename = 'FG validationset for tfrecords'
test_filename = 'FG testset for age.tfrecords'

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