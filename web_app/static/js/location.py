from flask import jsonify, request, Flask, session
from flask_pymongo import PyMongo



app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/mp2s"

mongo = PyMongo(app)


def save_location():
    if request.method == 'POST':
        data = request.get_json()
        # get username from session
        mongo.db['users'].update_one({"username": session['user'][0]}, {"$set" : {"location" : data}}, upsert=True)   