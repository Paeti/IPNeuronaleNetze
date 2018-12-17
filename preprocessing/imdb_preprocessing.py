import sys
import cv2 as cv
import tensorflow as tf
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import os.path

print("-> Starting read of metadata file")
with open('imdb_metadata.csv', 'r') as file:
    lines = [line.rstrip('\n') for line in file]
    # dob, full_path, gender, photo_taken, face_location
    dob = lines[0]
    full_path = lines[2]
    gender = lines[4]
    photo_taken = lines[6]
    face_location = lines[8]

    dob = dob.lstrip('[').rstrip(']')
    dob = dob.split(',')
    dob = [float(x) for x in dob]
    print(dob[0])

    full_path = full_path.split(' ')
    print(full_path[0])

    gender = gender.lstrip('[').rstrip(']')
    gender = gender.split(',')
    gender = [float(x) for x in gender]
    print(gender[0])

    photo_taken = photo_taken.lstrip('[').rstrip(']')
    photo_taken = photo_taken.split(',')
    photo_taken = [str(int(float(x))) for x in photo_taken]
    print(photo_taken[0])

    face_location = face_location.split(' ')
    face_location = [x.lstrip('[[').rstrip(']]').split(',') for x in face_location]
    print(face_location[0])
print("-> metadata file read complete, creating data arrays")

# maps file path to tuple containing the rest of the image information
I = {}
for idx, val in enumerate(full_path):
    try:
        I[val] = {"dob_m": dob[idx],
                  "dob": datetime.fromordinal(int(dob[idx])) + timedelta(days=int(dob[idx]) % 1) - timedelta(days=366),
                  "gender": gender[idx], "photo_taken": datetime.strptime(photo_taken[idx], "%Y"),
                  "face_location": face_location[idx]}
        I[val]["age"] = (I[val]["photo_taken"] - I[val]["dob"]).days / 365.2425
    except:
        I[val] = {"dob_m": dob[idx],
                  "dob": -1,
                  "gender": gender[idx], "photo_taken": datetime.strptime(photo_taken[idx], "%Y"),
                  "face_location": face_location[idx]}
        I[val]["age"] = -1
    I[val]["img_path"] = "imdb_crop/" + val

    if not os.path.isfile("imdb_crop" + "/" + val):
        del I[val]
        continue
    im_test = cv.imread("imdb_crop" + "/" + val)
    if im_test is None:
        del I[val]

#print(I["01/nm0000001_rm124825600_1899-5-10_1968.jpg"])

print("-> data array creating completed, flushing into training ready dataset")

X = []
Y_age = []
Y_gender = []
for k, v in I.items():
    X.append(v["img_path"])
    Y_age.append(
        v["age"]
    )
    Y_gender.append(
        v["gender"]
    )
print("-> training set ready for splitting")


#number of images for trainingset
size_training = (len(I) * 0.75) / len(I)

'''
    X_train = images for trainingset
    Y_age_train = age-labels for trainingset
    Y_gender_train = gender-labels for trainingset
    X_val = images for validationset
    Y_age_val = age-labels for validationset
    Y_gender_val = gender-labels for validationset
    X_test = images for testset
    Y_age_test = age-labels for testset
    Y_gender_test = gender-labels for testset
'''
#shuffle default = true
#stratify default = none --> nicht schichtenweise
X_train, X_tmp, Y_age_train, Y_age_tmp, Y_gender_train, Y_gender_tmp = train_test_split(
    X, Y_age, Y_gender, train_size=size_training, random_state=1
)

#number of images for validationset
size_val = (len(X_tmp) * 0.2) / len(X_tmp)

X_val, X_test, Y_age_val, Y_age_test, Y_gender_val, Y_gender_test = train_test_split(
    X_tmp, Y_age_tmp, Y_gender_tmp, train_size=size_val, random_state=1
)

print("-> dataset splitted")

#load images
def load_image(img_path):
    img = cv.imread(img_path)
    if img is None:
        return None
    img = cv.resize(img, (224, 224), interpolation=cv.INTER_CUBIC)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img


#convert data to features
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecord(path, datasetX, datasetY):
    import math
    writer = tf.python_io.TFRecordWriter(path)
    print("-> starting tfrecord export of " + path)
    
    #from tqdm import tqdm
    #for i in tqdm(range(len(datasetX))):
    for i in range(len(datasetX)):
        img = load_image(datasetX[i])
        if img is None:
            continue
        '''
        choose the right label you want to train on    
        '''

        label = datasetY[i]

        label = int(label) if not math.isnan(label) else -1

        feature = {'train/label': _int64_feature(label),
                   'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()

#write data in tfrecords file
train_filename1 = 'trainset for age.tfrecords'
train_filename2 = 'trainset for gender.tfrecords'
validation_filename1 = 'validationset for age.tfrecords'
validation_filename2 = 'validationset for gender.tfrecords'
test_filename1 = 'testset for age.tfrecords'
test_filename2 = 'testset for gender.tfrecords'

write_tfrecord(train_filename1, X_train, Y_age_train)
write_tfrecord(train_filename2, X_train, Y_gender_train)
write_tfrecord(validation_filename1, X_val, Y_age_val)
write_tfrecord(validation_filename2, X_val, Y_gender_val)
write_tfrecord(test_filename1, X_test, Y_age_test)
write_tfrecord(test_filename2, X_test, Y_age_test)

print("-> created and wrote trfrecords file for selected dataset")