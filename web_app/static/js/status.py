from flask import jsonify, request, Flask, session
from flask_pymongo import PyMongo
import datetime
import uuid
import bson



app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/mp2s"

mongo = PyMongo(app)

def update_status():
    if request.method == 'POST':
        print('LOGGED OUT')
        # get username from session
        mongo.db['users'].update_one({"username": session['user'][0]}, {"$set" : {"status" : False}}, upsert=True) 

def save_incident():
    # get username from session
    data = request.get_json()
    print(data)

    anomaly = data[data.find('=') + 1 : data.find('status') - 1] 
    status = data[data.find('status') + 7 : data.find('location') - 1]
    location = data[data.find('location') + 9 : data.find('officer_id') - 1]
    officer_id = data[data.find('officer_id') + 11 : data.find('start_time') - 1]
    start_time = data[data.find('start_time') + 11 : ]

    print('Incident Saved') 
    time = datetime.datetime.now()
    random_uuid = uuid.uuid4()
    mongo.db['incidents'].insert_one({"anomaly_class" : anomaly, "status" : status, "end_time" : time, "_id" :  bson.Binary.from_uuid(random_uuid), "start_time" : start_time, "camera_location" : location, "officer_username" : officer_id})
    
        
