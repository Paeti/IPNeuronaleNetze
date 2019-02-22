import os, sys, base64, re, requests, json
from flask import Flask, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from keras.applications.vgg16 import vgg16, preprocess_input
from keras.preprocessing import image
import numpy as np 


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
        print(os.getcwd())
        ID = getNextID()

        img_path = os.path.dirname(os.getcwd())+ "\\data\\production_img\\"+ str(ID) +"_.png"
        with open(img_path, 'wb') as f:
            f.write(base64.decodestring(base64String.split(',')[1].encode()))

        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        (age, gender) = getPrediction(x)
        return jsonify(age=age, gender=gender, id=ID)

@app.route('/save', methods=['GET', 'POST'])
def save():
    if request.method == 'POST':
        save = request.form['save']
        save = int(save)
        ID = request.form['ID']
        age = request.form['age']
        gender = request.form['gender']
        img_path = os.path.dirname(os.getcwd())+ "\\data\\production_img\\"+str(ID)+"_.png"
        if save == 0:
            os.remove(img_path)
        else:
            new_img_path = os.path.dirname(os.getcwd())+ "\\data\\production_img\\"+str(ID)+"_"+str(age)+"_"+str(gender)+".png"
            os.rename(img_path, new_img_path)
        return jsonify(success=True)
        
def getPrediction(x):
    # headers = {"content-type": "application/json"}
    # data = json.dumps({"signature_name": "serving_default", "instances": x.tolist()})
    # age_response = requests.post('http://localhost:9000/v1/models/Age:predict', data=data, headers=headers)
    # gender_response = requests.post('http://localhost:9001/v1/models/Gender:predict', data=data, headers=headers)
    
    # age_prediction = json.loads(age_response.text)['predictions']
    # #print(age_prediction)
    # gender_prediction = json.loads(gender_response.text)['predictions']
    # #print(gender_prediction)

    # age = age_prediction[0].index(max(age_prediction[0]))
    # gender = "M"
    # if gender_prediction[0][0] < 0.5:
    #     gender = "W"

    # return (age, gender)
    return (100, "T")

def getNextID():
    path = os.path.dirname(os.getcwd())+ "\\data\\production_img\\"
    dirs = os.listdir(path)
    nextID = 0
    for file in dirs:
        ID = int(file.split('_')[0])
        if ID > nextID:
            nextID = ID
    return nextID + 1

if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0')