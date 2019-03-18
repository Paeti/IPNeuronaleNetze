import urllib.request
import os
import csv
import cv2 as cv
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split


class Dataset:

    face_cascade = cv.CascadeClassifier('../data_loader/haarcascade_frontalface_default.xml')

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


    def downloadAndUnpack(self, url, target_zip_directory, target_extract_directory):
        if not os.path.exists(target_zip_directory):
            print("Start downloading Tarfile")
            urllib.request.urlretrieve(url, target_zip_directory)
            print("Finished downloading Tarfile")
        else:
            print("Tarfile already downloaded.")

        if not os.path.exists(target_extract_directory):
            print("Start unpacking")
            import tarfile
            tar = tarfile.open(target_zip_directory)
            tar.extractall(path="../data")
            tar.close()
            print("Finished unpacking")
        else:
            print("Already unpacked")


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
           tmp_fix = ""
           fix = a.split('/')
        for idx, fix_1 in enumerate(fix):
            tmp_fix = tmp_fix + fix_1
            if idx == 4:
                tmp_fix = tmp_fix + "\\"
            elif idx < 5:
                tmp_fix = tmp_fix + "/"
        
            b = tmp_fix.split('\\')
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

    def readAndPrintDataImagesIMDB(self, csv_directory, image_directory, classification_folder):
        with open(csv_directory, 'r') as file:
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
                          "dob": datetime.fromordinal(int(dob[idx])) + timedelta(days=int(dob[idx]) % 1) - timedelta(
                              days=366),
                          "gender": gender[idx], "photo_taken": datetime.strptime(photo_taken[idx], "%Y"),
                          "face_location": face_location[idx]}
                I[val]["age"] = (I[val]["photo_taken"] - I[val]["dob"]).days / 365.2425
            except:
                I[val] = {"dob_m": dob[idx],
                          "dob": -1,
                          "gender": gender[idx], "photo_taken": datetime.strptime(photo_taken[idx], "%Y"),
                          "face_location": face_location[idx]}
                I[val]["age"] = -1
            I[val]["img_path"] = image_directory + "/" + val

            if not os.path.isfile(image_directory + "/" + val):
                del I[val]
                continue
            # im_test = cv.imread("../data/imdb_crop" + "/" + val)
            # if im_test is None:
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

        # number of images for trainingset
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
        # shuffle default = true
        # stratify default = none --> nicht schichtenweise
        X_train, X_tmp, Y_age_train, Y_age_tmp, Y_gender_train, Y_gender_tmp = train_test_split(
            X, Y_age, Y_gender, train_size=size_training, random_state=1
        )

        # number of images for validationset
        size_val = (len(X_tmp) * 0.2) / len(X_tmp)

        X_val, X_test, Y_age_val, Y_age_test, Y_gender_val, Y_gender_test = train_test_split(
            X_tmp, Y_age_tmp, Y_gender_tmp, train_size=size_val, random_state=1
        )

        self.print_images(X_train, Y_age_train, "age/Train", classification_folder)
        self.print_images(X_train, Y_gender_train, "gender/Train", classification_folder)
        self.print_images(X_val, Y_age_val, "age/Valid", classification_folder)
        self.print_images(X_val, Y_gender_val, "gender/Valid", classification_folder)
        self.print_images(X_test, Y_age_test, "age/Test", classification_folder)
        self.print_images(X_test, Y_gender_test, "gender/Test", classification_folder)

        print("-> dataset splitted")

    def print_images(self, datasetX, datasetY, t, classification_folder):
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

            faces = self.face_cascade.detectMultiScale(img, 1.8, 5)
            for (x, y, w, h) in faces:
                cip = img[y - 5:y + h + 5, x - 5:x + w + 5].copy()
                try:
                    cip = cv.resize(cip, (224, 224))
                    counter += 1
                    if not os.path.exists(classification_folder + "/" + t + "/" + str(label)):
                        os.makedirs(classification_folder + "/" + t + "/" + str(label))
                    cv.imwrite(
                        classification_folder + "/" + t + "/" + str(label) + "/" + str(counter) + "_" + datasetX[i].split("/")[
                            -1], cip)
                    # print("W " + "../data/IMDB/" + t + "/" + str(int(label)) + "/" + str(counter) + "_" + datasetX[i].split("/")[-1])
                except:
                    print("Resize Fail")
'''

dataset = Dataset()
dataset.createFolder("../data/classification")
dataset.createClassificationFolders("../data/classification/age/Train")
dataset.createClassificationFolders("../data/classification/age/Valid")
dataset.createClassificationFolders("../data/classification/age/Test")
#LAP
dataset.downloadAndUnzip("http://158.109.8.102/AppaRealAge/appa-real-release.zip", "../data/appa-real-release.zip", "../data/appa-real-release")
dataset.readAndPrintDataImagesLAP("../data/appa-real-release", "train", "../data/classification/age/Train")
dataset.readAndPrintDataImagesLAP("../data/appa-real-release", "valid", "../data/classification/age/Valid")
dataset.readAndPrintDataImagesLAP("../data/appa-real-release", "test", "../data/classification/age/Test")

#FGNET
dataset.downloadAndUnzip("http://yanweifu.github.io/FG_NET_data/FGNET.zip", "../data/FGNET.zip", "../data/FGNET")
dataset.readAndPrintDataImagesFGNET("../data/FGNET", "../data/classification/age")
#IMDB
dataset.downloadAndUnpack("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb"
                         "_crop.tar", "../data/imdb_crop.tar", "../data/imdb_crop");
dataset.readAndPrintDataImagesIMDB('../data/imdb_metadata.csv', "../data/imdb_crop", "../data/classification")

'''
