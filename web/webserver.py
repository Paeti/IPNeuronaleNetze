from flask import Flask
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

class AskNN(Resource):
    def post(self, id, permission):
        pass

class LoadPage(Resource):
    def get(self):
        return make_response(render_template('index.html'),200,headers)


api.add_resource(LoadPage, '/', endpoint = 'index')
api.add_resource(AskNN, '/predict', endpoint = 'predict')

app.run(debug = True)
