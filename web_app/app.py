import cv2
from flask import Flask, render_template, Response, make_response, redirect, url_for, session, request, jsonify
from functools import wraps
from utils.util import get_model1, get_model2, get_tensor, extract_flowMap
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.vote import majority_voting
import json
from time import time
from random import random
import requests
import sys
from flask_pymongo import PyMongo
from utils.compute_distance import minimum_distance
from user.models import User
from static.js.location import save_location
from static.js.status import update_status
from static.js.status import save_incident
from pywebpush import webpush, WebPushException
from flask_socketio import SocketIO, emit
from threading import Lock
import datetime
import os

thread = None
thread_lock = Lock()

app = Flask(__name__)
app.secret_key = b'\xb0Rh\x80\xa4|\r\xbes\x87b\xaf\xf2\xd1\x9c3'
# socketio = SocketIO(app, cors_allowed_orgins="")

app.config["MONGO_URI"] = "mongodb://localhost:27017/mp2s"

mongo = PyMongo(app)


camera = cv2.VideoCapture('static/test2.mp4')
# camera = cv2.VideoCapture('static/test1.mp4') 
# camera = cv2.VideoCapture('static/test2.mp4')
# camera = cv2.VideoCapture('static/test3.mp4')   
# camera = cv2.VideoCapture(1) 


def reconstruction_loss(label, prediction): 
    # change to custom loss function if desired 
    with torch.no_grad():
        return torch.nn.MSELoss()(prediction, label)

frame_list, loss_list, prob_list, vote_list = [0], [0], [0], [False]
def generate_frames():
    '''
        1. Send 8 consecutive frames to the model
        2. Retrieve predictions for these frames (reconstruction loss value, prediction_probability) and call generate_graphs function
        3. Call majority voting function
        4. Call notification function - if majority voting calls for it
    '''

    count = 1
    image_list = []
    while True:
        success, image = camera.read()
        if not success: 
            break
        else:
            ret, buffer = cv2.imencode('.jpg', image) 
            frame = buffer.tobytes()
            image_list.append(image)
            
            model1 = get_model1()
            model2 = get_model2()
            
            if count % 8 == 0:
                flowMap = extract_flowMap(image_list)
                image_list = []

                image_tensor = get_tensor(flowMap).permute(1,0,2,3)
                prediction = model1(image_tensor) # feed in the flow map not the image

                # compute reconstruction loss
                loss = reconstruction_loss(image_tensor, prediction) 
                loss_list.pop()
                loss_list.append(loss.item())

                # compute classification probability
                image_tensor = get_tensor(image)
                cls_probability = torch.sigmoid(model2(image_tensor)).tolist()
                prob_list.pop() 
                prob_list.append(cls_probability[0][0])

                # compute majority voting here
                vote = majority_voting(loss, cls_probability) 
                vote_list.pop()
                vote_list.append(vote) 
                

                
            count+=1 
            frame_list.pop()
            frame_list.append(count)


            yield(b'--frame\r\n'
                    b'Content-type:image/jpeg\r\n\r\n' + frame + b'\r\n') 
            

# Decorators
def login_required(f):
  @wraps(f)
  def wrap(*args, **kwargs):
    if 'logged_in' in session:
      return f(*args, **kwargs)
    else:
      return redirect('/')
  
  return wrap


@app.route('/data/', methods=['GET', 'POST']) 
@login_required
def generate_data(): 
    data = [time() * 1000, loss_list[0], prob_list[0]] 
    response = make_response(json.dumps(data))
    response.content_type = 'application/json' 
    return response  


@app.route('/notify/', methods=['GET', 'POST'])  
@login_required
def notifier(): 
    data = [vote_list[0]]  
    response = make_response(json.dumps(data))
    response.content_type = 'application/json' 
    return response  
  

@app.route('/dashboard/')
@login_required
def index():
    return render_template('index.html')


@app.route('/')
def login():
    return render_template('login.html')


@app.route('/officer_notification/')
@login_required
def officer_notification():

    # use session to retrieve all required info from database
    user = session['user']
    notification = mongo.db['notifications'].find_one({"officer_id" : user[0]}) 
    print('Reloaded')
    if notification is not None:
        notify = True
    else:
        notify = False
    return jsonify(notify)

@app.route('/officer_page/')
@login_required
def officer():

    # use session to retrieve all required info from database
    user = session['user']
    cursor = mongo.db['users'].find_one({"username" : user[0]})  
    notification = mongo.db['notifications'].find_one({"officer_id" : user[0]})  
    return render_template('officer_page.html', location= cursor['location'], username=cursor['username'], notification=notification) 



@app.route('/user/login/', methods=['GET', 'POST'])
def login_user():
    return User().login() 


@app.route('/user/logout/', methods=['GET', 'POST'])
def logout():
    return User().logout()

@app.route('/compute_distance', methods=['GET', 'POST'])
@login_required
def distance():
    minimum_distance()
    return render_template('index.html') 


@app.route('/location', methods=['GET', 'POST'])
@login_required
def location():
    save_location()
    return render_template('officer.html', data=[])


@app.route('/status/', methods=['GET', 'POST'])
@login_required
def status():
    update_status()
    return render_template('officer.html', data=[])


@app.route('/incident/', methods=['POST'])
@login_required
def incident():
    user = session['user']
    cursor = mongo.db['users'].find_one({"username" : user[0]})  
    notification = mongo.db['notifications'].find_one({"officer_id" : user[0]})  

    if request.method == 'POST':
        save_incident()
    return render_template('officer_page.html', location= cursor['location'], username=cursor['username'], notification=notification)  


@app.route('/video')
@login_required
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/officer/', methods=['GET', 'POST']) 
@login_required
def notification():
    print('REACHED') 
    user = session['user']
    # for listening
    cursor = mongo.db['notifications'].find_one({"officer_id" : user[0]}) 
    # If cursor is not None, alert the officer with the above user id
    location = []
    officer_id = 0
    start_time = 0
    notify = False

    if cursor is not None:
        location = cursor['location']
        officer_id = cursor['officer_id']
        start_time = cursor['start_time']
        notify = True 
        # return render_template('officer.html', data=[location, officer_id, start_time, notify]) 
    
    # delete notification corresponding to the user since that expired    
    mongo.db['notifications'].delete_one({"officer_id" : user[0]}) 
    return render_template('officer.html', data=[location, officer_id, start_time, notify]) 



# def background_thread():
#     """Example of how to send server generated events to clients."""
#     count = 0
#     while True:
#         socketio.sleep(10) 
#         count += 1
#         socketio.emit('response',
#                       {'data': 'Server generated event'})


# @socketio.on('event')
# def send_notification(message):
#     print('Event Sent')
#     emit('response', {'data': message['data']})


# @socketio.on('disconnect')
# def disconnect():
#     print('Client disconnected')


if __name__== '__main__':
    app.config['SERVER_NAME'] = "127.0.0.1:5000"
    # flask-socketio cannot handle running the models - find a fix to that but for now query the database every few seconds for updates
    # socketio.run(app)
    app.run(debug=True)

