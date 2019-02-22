import urllib.request
import os
import csv
import cv2 as cv
import random

class Dataset:

    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    def createFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print("Error: Creating directory.")

    def createClassificationFolders(self, directory):
        self.createFolder(directory)
        for x in range(101):
            self.createFolder(directory + "/" + str(x))

    def downloadAndUnzip(self, url, target_zip_directory, target_extract_directory):
        if not os.path.exists(target_zip_directory):
            print("Start downloading Zipfile")
            urllib.request.urlretrieve(url, target_zip_directory)
            print("Finished downloading Zipfile")
        else:
            print("Zip already downloaded.")

        if not os.path.exists(target_extract_directory):
            print("Start unzipping")
            import zipfile
            with zipfile.ZipFile(target_zip_directory, 'r') as zip_ref:
                zip_ref.extractall(target_extract_directory)
            print("Finished unzipping")
        else:
            print("Already unzipped")

    def readAndPrintDataImagesLAP(self, image_directory, type_set_csv, classification_target_directory):
        age = []
        full_path =[]
        delta = 5
        with open(image_directory + "/appa-real-release/gt_avg_" + type_set_csv + ".csv", "r") as f:
            reader = csv.reader(f, delimiter="\\")
            for i, line in enumerate(reader):
                if i != 0:
                    age_string = line[0].split(',')[4]
                    age_float = float(age_string)
                    age.append(age_float)
                    path = line[0].split(',')[0]
                    full_path.append(path)

        for idx, val in enumerate(full_path):
            img = cv.imread(image_directory + "/appa-real-release/" + type_set_csv + "/" + val)
            faces = self.face_cascade.detectMultiScale(img, 1.8, 5)
            for (x,y,w,h) in faces:
                cip = img[y-delta:y+h+delta, x-delta:x+w+delta].copy()
                try:
                    cip = cv.resize(cip, (224, 224))
                    cv.imwrite(classification_target_directory + "/" + str(int(age[idx])) + "/" + val, cip)
                    print("Written: " + classification_target_directory + "/" + str(int(age[idx])) + "/" + val)
                except:
                    print("Resize Fail")

    def load_images_from_folder(self, folder):
        images = []
        for subfolder in os.listdir(folder):
            images.append(os.path.join(folder, subfolder))
        return images

    def readAndPrintDataImagesFGNET(self, image_directory, classification_target_directory):
        age = []
        full_path = []
        full_path2 = self.load_images_from_folder(image_directory + "/FGNET/images")
        delta = 5

        for a in full_path2:
            b = a.split('\\')
            full_path.append(b[1])
            b = b[1].split('.')

            c = b[0].split('A')
            d = c[1].strip('a')
            d = d.strip('b')
            age.append(d)

        print("Printing Images to Classification Folders")

        for idx, val in enumerate(full_path):
            img = cv.imread(image_directory + "/FGNET/images/" + val)
            faces = self.face_cascade.detectMultiScale(img, 1.8, 5)
            for (x, y, w, h) in faces:
                cip = img[y - delta:y + h + delta, x - delta:x + w + delta].copy()
                try:
                    cip = cv.resize(cip, (224, 224))
                    i = random.random()
                    if i > 0.5:
                        cv.imwrite(classification_target_directory + "/Train/" + str(int(age[idx])) + "/" + val, cip)
                        print("Written: " + classification_target_directory + "/Train/" + str(int(age[idx])) + "/" + val)
                    elif i < 0.25:
                        cv.imwrite(classification_target_directory + "/Valid/" + str(int(age[idx])) + "/" + val, cip)
                        print("Written: " + classification_target_directory + "/Valid/" + str(int(age[idx])) + "/" + val)
                    else:
                        cv.imwrite(classification_target_directory + "/Test/" + str(int(age[idx])) + "/" + val, cip)
                        print("Written: " + classification_target_directory + "/Test/" + str(int(age[idx])) + "/" + val)
                except:
                    print("Resize Fail")

dataset = Dataset()
dataset.createFolder("../data/classification")
dataset.createClassificationFolders("../data/classification/Train")
dataset.createClassificationFolders("../data/classification/Valid")
dataset.createClassificationFolders("../data/classification/Test")
#LAP
dataset.downloadAndUnzip("http://158.109.8.102/AppaRealAge/appa-real-release.zip", "../data/appa-real-release.zip", "../data/appa-real-release")
dataset.readAndPrintDataImagesLAP("../data/appa-real-release", "train", "../data/classification/Train")
dataset.readAndPrintDataImagesLAP("../data/appa-real-release", "valid", "../data/classification/Valid")
dataset.readAndPrintDataImagesLAP("../data/appa-real-release", "test", "../data/classification/Test")

#FGNET
dataset.downloadAndUnzip("http://yanweifu.github.io/FG_NET_data/FGNET.zip", "../data/FGNET.zip", "../data/FGNET")
dataset.readAndPrintDataImagesFGNET("../data/FGNET", "../data/classification")