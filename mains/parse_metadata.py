import sys
import os
from datetime import datetime, timedelta

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
    I[val]["img"] = cv.imread("imdb_crop" + "/" + val)
    I[val]["img"] = cv.resize(I[val]["img"], (224, 224))
    #TODO check if color correcting does something

print(I["01/nm0000001_rm124825600_1899-5-10_1968.jpg"])

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
