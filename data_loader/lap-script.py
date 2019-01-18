import urllib
import os
import csv
import cv2 as cv

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

if not os.path.exists("../data/appa-real-release.zip"):
    print("Start downloading Zipfile")
    urllib.urlretrieve("http://158.109.8.102/AppaRealAge/appa-real-release.zip", "../data/appa-real-release.zip")
    print("Finished downloading Zipfile")
else:
    print("Zip already downloaded.")

if not os.path.exists("../data/appa-real-release"):
    print("Start unzipping")
    import zipfile
    with zipfile.ZipFile("../data/appa-real-release.zip", 'r') as zip_ref:
        zip_ref.extractall("../data/appa-real-release")
    print("Finished unzipping")
else:
    print("Already unzipped")


folder_directory = "../data/"

if not os.path.exists(folder_directory + "LAP"):
    print("Start creating Classification folders")
    createFolder(folder_directory + "LAP")
    createClassificationFolders(folder_directory + "LAP/Train")
    createClassificationFolders(folder_directory + "LAP/Valid")
    createClassificationFolders(folder_directory + "LAP/Test")
    print("Finished creating Classification folders")
else:
    print("Classification folders already present")


def readAndPrintDataImages(type_set_csv, type_set_target):
    age = []
    full_path =[]
    delta = 5
    counter = 0
    with open("../data/appa-real-release/appa-real-release/gt_avg_" + type_set_csv + ".csv", "r") as f:
        reader = csv.reader(f, delimiter="\\")
        for i, line in enumerate(reader):
            if i != 0:
                age_string = line[0].split(',')[4]
                age_float = float(age_string)
                age.append(age_float)
                path = line[0].split(',')[0]
                full_path.append(path)


    for idx, val in enumerate(full_path):
        img = cv.imread("../data/appa-real-release/appa-real-release/" + type_set_csv + "/" + val)
        faces = face_cascade.detectMultiScale(img, 1.8, 5)
        for (x,y,w,h) in faces:
            cip = img[y-delta:y+h+delta, x-delta:x+w+delta].copy()
            try:
                cip = cv.resize(cip, (224, 224))
                counter = counter + 1
                cv.imwrite("../data/LAP/" + type_set_target + "/" + str(int(age[idx])) + "/" + val, cip)
                print("Written: " + "../data/LAP/" + type_set_target + "/" + str(int(age[idx])) + "/" + val)
            except:
                print("Resize Fail")
    return counter

print("Printing Images to Classification Folders")
train_img_count = readAndPrintDataImages("train", "Train")
valid_img_count = readAndPrintDataImages("valid", "Valid")
test_img_count = readAndPrintDataImages("test", "Test")

f = open("../data/img_counts.txt", "w+")
f.write("Train_img_count = " + str(train_img_count) + "\n")
f.write("Validation_img_count = " + str(valid_img_count) + "\n")
f.write("Test_img_count = " + str(test_img_count) + "\n")
f.close()

print("FINISHED")
