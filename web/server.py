from flask import Flask, render_template
app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict(id, permission):
    return 'Hello World!!!!'

if __name__ == '__main__':
  app.run(host = '0.0.0.0', debug = True)
