from flask import Flask, jsonify, session, redirect, request
from bson import json_util  
from flask_pymongo import PyMongo

import json

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/mp2s"

mongo = PyMongo(app)

class User:

    def start_session(self, user):
        session['logged_in'] = True
        session["user"] = user
        return jsonify(user[0], user[1]), 200

    
    def logout(self):
        session.clear()
        return redirect('/') 
    

    def login(self):
        user = mongo.db['users'].find_one({
            "username" : request.form.get('email')
        })

        # if mongo.db['users'].find_one({"role" : "officer"}):
        #     mongo.db['users'].update_one({"username": session['user'][0]}, {"$set" : {"status" : True}}, upsert=True) 

        if user and (request.form.get("password") == user["password"]):
            mongo.db['users'].update_one({"username": user['username']}, {"$set" : {"status" : True}}, upsert=True) 
            return self.start_session([user['username'], user['role']]) 

        return jsonify({ "error" : "Invalid login credentials" }), 401
     