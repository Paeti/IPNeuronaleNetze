from flask import Flask
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

class AskNN(Resource):
    def post(self, id, permission):
        pass

class LoadPage(Resource):
    def get(self):
        pass


api.add_resource(UserAPI, '/', endpoint = 'index')
api.add_resource(UserAPI, '/app', endpoint = 'app')
