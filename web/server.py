import os, sys, base64, re, time
from PIL import Image
from io import BytesIO
from flask import Flask, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from keras.applications.vgg16 import vgg16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np 


UPLOAD_FOLDER = '/pictures'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#STYLE = url_for('static', filename='style.css')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        base64String = request.form['image']
        """imageBinary = base64.b64decode(base64String + "===")
        ID = getNextID()
        img_path = os.getcwd()+ "\\bilder\\"+ str(ID) +"_.png"

        with open(img_path, 'wb') as f:
            f.write(imageBinary)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)"""
        ID = getNextID()
        img_path = os.getcwd()+ "\\bilder\\"+ str(ID)
        getI420FromBase64(base64String +"===",img_path)

        (age, gender) = getPrediction(x)
        return jsonify(age=age, gender=gender, id=ID)

@app.route('/save', methods=['GET', 'POST'])
def save():
    if request.method == 'POST':
        save = request.form['save']
        ID = request.form['ID']
        age = request.form['age']
        gender = request.form['gender']
        img_path = os.getcwd()+ "/bilder/"+ID+"_.jpg"
        if save == False:
            os.remove(img_path)
        else:
            new_img_path = os.getcwd()+ "/bilder/"+ID+"_"+age+"_"+gender+".jpg"
            os.rename(img_path, new_img_path)
        return jsonify(success=True)
        
def getPrediction(x):
    #send to dockercontainer, get result
    #filename = secure_filename(file.filename)
    #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    (age, gender) = 10, 'M'
    return (age, gender)

def getNextID():
    path = os.getcwd()+ "/bilder"
    dirs = os.listdir(path)
    nextID = 0
    for file in dirs:
        ID = int(file.split('_')[0])
        if ID > nextID:
            nextID = ID
    return nextID + 1

def getI420FromBase64(codec, image_path="c:\\"):
    base64_data = re.sub('^data:image/.+;base64,', '', codec)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    t = time.time()
    img.save(image_path + str(t) + '.png', "PNG")