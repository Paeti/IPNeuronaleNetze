import urllib
import os
import csv
import cv2 as cv
import random
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import os.path


face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory.")

def createClassificationFolders(directory):
    createFolder(directory)
    for x in range(101):
        createFolder(directory + "/" + str(x))

def load_images_from_folder(folder):
    images = []
    for subfolder in os.listdir(folder):
        images.append(os.path.join(folder, subfolder))
    return images

if not os.path.exists("../data/imdb_crop.tar"):
    print("Start downloading Tarfile")
    urllib.urlretrieve("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar", "../data/imdb_crop.tar")
    print("Finished downloading Tarfile")
else:
    print("Tarfile already downloaded.")

if not os.path.exists("../data/imdb_crop"):
    print("Start unpacking")
    import tarfile
    tar = tarfile.open("../data/imdb_crop.tar")
    tar.extractall(path="../data")
    tar.close()
    print("Finished unpacking")
else:
    print("Already unpacked")

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
    I[val]["img_path"] = "../data/imdb_crop/" + val

    if not os.path.isfile("../data/imdb_crop" + "/" + val):
        del I[val]
        continue
    #im_test = cv.imread("../data/imdb_crop" + "/" + val)
    #if im_test is None:
    #    del I[val]

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
print(X[0])
print(Y_age[0])
print(Y_gender[0])

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

def write_tfrecord(datasetX, datasetY, t):
    counter = 0
    import math
    print("-> starting export of " + t)
    
    for i in range(len(datasetX)):
        img = cv.imread(datasetX[i])
        if img is None:
            	continue
        label = datasetY[i]
        label = int(label) if not math.isnan(label) else -1
        if label < 0 or label > 100:
                continue

        faces = face_cascade.detectMultiScale(img, 1.8, 5)
        for (x,y,w,h) in faces:
                cip = img[y-5:y+h+5, x-5:x+w+5].copy()
                try:
                      cip = cv.resize(cip, (224, 224))
                      counter += 1
                      if not os.path.exists("../data/IMDB/" + t + "/" + str(label)):
                            os.makedirs("../data/IMDB/" + t + "/" + str(label))
                      cv.imwrite("../data/IMDB/" + t + "/" + str(label) + "/" + str(counter) + "_" + datasetX[i].split("/")[-1], cip)
                      #print("W " + "../data/IMDB/" + t + "/" + str(int(label)) + "/" + str(counter) + "_" + datasetX[i].split("/")[-1])
                except:
                      print("Resize Fail")


print(X_train[X_train.index(X[0])])
print(Y_age_train[X_train.index(X[0])])
print(Y_gender_train[X_train.index(X[0])])

write_tfrecord(X_train, Y_age_train, "age/train")
write_tfrecord(X_train, Y_gender_train, "gender/train")
write_tfrecord(X_val, Y_age_val, "age/val")
write_tfrecord(X_val, Y_gender_val, "gender/val")
write_tfrecord(X_test, Y_age_test, "age/test")
write_tfrecord(X_test, Y_gender_test, "gender/test")

print("-> created and wrote trfrecords file for selected dataset")

'''
age = []
full_path = []
full_path2 = load_images_from_folder("../data/FGNET/FGNET/images")
delta = 5

for a in full_path2:
    b = a.split('\\')
    full_path.append(b[1])
    b = b[1].split('.')

    c = b[0].split('A')
    d = c[1].strip('a')
    d = d.strip('b')
    age.append(d)

counter_training = 0
counter_valid = 0
counter_test = 0
print("Printing Images to Classification Folders")

for idx, val in enumerate(full_path):
    img = cv.imread("../data/FGNET/FGNET/images/" + val)
    faces = face_cascade.detectMultiScale(img, 1.8, 5)
    for (x,y,w,h) in faces:
        cip = img[y-delta:y+h+delta, x-delta:x+w+delta].copy()
        try:
            cip = cv.resize(cip, (224, 224))
            i = random.random()
            if i > 0.5:
                counter_training = counter_training + 1
                cv.imwrite("../data/LAP/Train/" + str(int(age[idx])) + "/" + val, cip)
                print("Written: " + "../data/LAP/Train/" + str(int(age[idx])) + "/" + val)
            elif i < 0.25:
                counter_valid = counter_valid + 1
                cv.imwrite("../data/LAP/Validation/" + str(int(age[idx])) + "/" + val, cip)
                print("Written: " + "../data/LAP/Validation/" + str(int(age[idx])) + "/" + val)
            else:
                counter_test = counter_test + 1
                cv.imwrite("../data/LAP/Test/" + str(int(age[idx])) + "/" + val, cip)
                print("Written: " + "../data/LAP/Test/" + str(int(age[idx])) + "/" + val)
        except:
            print("Resize Fail")

f = open("../data/img_counts_fg.txt", "w+")
f.write("Train_img_count = " + str(counter_training) + "\n")
f.write("Validation_img_count = " + str(counter_valid) + "\n")
f.write("Test_img_count = " + str(counter_test) + "\n")
f.close()
'''
