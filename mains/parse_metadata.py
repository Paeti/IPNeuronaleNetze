import sys
import cv2 as cv
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

print("-> Starting read of metadata file")
with open('metadata.csv', 'r') as file:
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
#changing path to imdb file !!
    I[val]["img"] = cv.imread("imdb" + "/" + val)
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
    Y.append((
        v["age"],
        v["gender"]
    ))
print("-> training set ready for splitting")


#size_training = 300000/len(I)
#size_test = 10000/len(I)

size_training = 300000/len(I)

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

size_val = 90000/len(X_tmp)

X_val, X_test, Y_val, Y_test = train_test_split(
    X_tmp, Y_tmp, train_size=size_val, random_state=1
)

