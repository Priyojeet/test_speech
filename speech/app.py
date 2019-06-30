from finalizing import makePred
from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)


@app.route('/prediction')
def index():
    data = makePred()
    return data


if __name__ == '__main__':

    app.run()