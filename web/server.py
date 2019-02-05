import os
from flask import Flask, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename

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
        file = request.files['picture']
        (age, gender) = getPrediction(file)
        return jsonify(age=age, gender=gender)

@app.route('/save', methods=['GET', 'POST'])
def save():
    if request.method == 'POST':
        save = request.form['save']
        ID = request.form['ID']
        if save == False:
            deletePicture(ID)
    return render_template('enddialog.html')
        
def getPrediction(file):
    #send to dockercontainer, get result
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    (age, gender) = None, None
    return (age, gender)

def deletePicture(ID):
    #delete Picture
    return None
