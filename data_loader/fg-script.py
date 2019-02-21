import urllib.request
import os
import csv
import cv2 as cv
import random

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

if not os.path.exists("../data/FGNET.zip"):
    print("Start downloading Zipfile")
    urllib.request.urlretrieve("http://yanweifu.github.io/FG_NET_data/FGNET.zip", "../data/FGNET.zip")
    print("Finished downloading Zipfile")
else:
    print("Zip already downloaded.")

if not os.path.exists("../data/FGNET"):
    print("Start unzipping")
    import zipfile
    with zipfile.ZipFile("../data/FGNET.zip", 'r') as zip_ref:
        zip_ref.extractall("../data/FGNET")
    print("Finished unzipping")
else:
    print("Already unzipped")


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

print("FINISHED")