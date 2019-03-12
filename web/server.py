import os, sys, base64, re, requests, json
from flask import Flask, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from keras.applications.vgg16 import vgg16, preprocess_input
from keras.preprocessing import image
import numpy as np
import cv2 as cv


UPLOAD_FOLDER = '/uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        base64String = request.form['image']
        base64String += "==="
        ID = getNextID()
        img_path = os.path.dirname(os.getcwd())+ "/data/production_img/"+ str(ID) +"_.png"
        with open(img_path, 'wb') as f:
            f.write(base64.decodestring(base64String.split(',')[1].encode()))

        face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        img = cv.imread(img_path)
        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img, 1.7, 5)
        print("--------------------------------------------")
        print(len(faces))
        width = 0
        delta = 0
        for (x,y,w,h) in faces:
            print(x,y,w,h)
            if w > width:
                width = w
                cip = img[y-delta:y+h+delta, x-delta:x+w+delta].copy()
                try:
                    cip = cv.resize(cip, (224, 224))
                    cv.imwrite(img_path, cip)
                    print("Written: " + img_path)
                except:
                    print("Resize Fail")

        if width > 0:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            (age, gender) = getPrediction(x)
            return jsonify(age=age, gender=gender, id=ID)
        else:
            os.remove(img_path)
            return jsonify(age=0, gender="M", id=0)

@app.route('/save', methods=['GET', 'POST'])
def save():
    if request.method == 'POST':
        save = request.form['save']
        save = int(save)
        ID = request.form['ID']
        age = request.form['age']
        gender = request.form['gender']
        img_path = os.path.dirname(os.getcwd())+ "/data/production_img/"+str(ID)+"_.png"
        if save == 0:
            try:
                os.remove(img_path)
            except:
                print("Already removed")
        else:
            new_img_path = os.path.dirname(os.getcwd())+ "/data/production_img/"+str(ID)+"_"+str(age)+"_"+str(gender)+".png"
            os.rename(img_path, new_img_path)
        return jsonify(success=True)

def getPrediction(x):
    headers = {"content-type": "application/json"}
    data = json.dumps({"signature_name": "serving_default", "instances": x.tolist()})
    age_response = requests.post('http://age-model-service:8501/v1/models/Age:predict', data=data, headers=headers)
    gender_response = requests.post('http://gender-model-service:8501/v1/models/Gender:predict', data=data, headers=headers)

    # age_response = requests.post('http://localhost:8502/v1/models/Age:predict', data=data, headers=headers)
    # gender_response = requests.post('http://localhost:8503/v1/models/Gender:predict', data=data, headers=headers)

    age_prediction = json.loads(age_response.text)['predictions']
    # print(age_prediction)
    gender_prediction = json.loads(gender_response.text)['predictions']
    # print(gender_prediction)

    age = age_prediction[0].index(max(age_prediction[0]))
    gender = "M"
    if gender_prediction[0][0] < 0.5:
        gender = "W"

    return (age, gender)
    # return (100, "T")

def getNextID():
    path = os.path.dirname(os.getcwd())+ "/data/production_img"
    dirs = os.listdir(path)
    nextID = 0
    for file in dirs:
        if file != ".keep":
            ID = int(file.split('_')[0])
            if ID > nextID:
                nextID = ID
    return nextID + 1

if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0', ssl_context = "adhoc")
