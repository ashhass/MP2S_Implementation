from flask import Flask, request
from flask_pymongo import PyMongo
import requests, json
import datetime
import geopy.distance

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/mp2s"

def compute_distance(array, mongo):
    source = str(array[0])
    # for i in range(1, len(array)):
    #     dest = str(array[i])
    #     api_key = 'AIzaSyA-F0YlMhOUj7O4HeTaHJmkfwcPjpdWQU8'
    #     url ='https://maps.googleapis.com/maps/api/distancematrix/json?'
    #     r = requests.get(url + 'origins=' + source +
    #                 '&destinations=' + dest +
    #                 '&key=' + api_key) 
    #     print(url + 'origins=' + source +
    #                 '&destinations='+ dest +
    #                 '&key=' + api_key)
    #     x = r.json()

    # compute distance
    distance = []
    # location1 = distance.append((array[1], geopy.distance.geodesic(array[0], array[1])))
    # location2 = distance.append((array[2], geopy.distance.geodesic(array[0], array[2])))
    # location3 = distance.append((array[3], geopy.distance.geodesic(array[0], array[3])))
    # location4 = distance.append((array[4], geopy.distance.geodesic(array[0], array[4])))

    # sorted_distance = sorted(distance[1]) 
    # print(sorted_distance[0]) 

    officer = mongo.db['users'].find({"role" : "officer", "location" : array[1]})


    # Google Maps not allowing to change to IP address restriction so switching to calculating location based on lat and long temporarily
    
    location = mongo.db['cameras'].find_one({'camera' : "camera_2"})['location'] 

    for element in officer:
        id = element['username']
    notify_officer(id, location, mongo) 


def notify_officer(officer, location, mongo):

    # save notification to database along with start time, camera location and officer id
    time = datetime.datetime.now()
    mongo.db['notifications'].insert_one({"location": location, "officer_id" : officer, "start_time" : time})
    print('Notification added') 



def minimum_distance():
    mongo = PyMongo(app)

    cursor = mongo.db['users'].find({"role" : "officer", "status" : True}) 
    print('Called')
    location_array = []

    # retrieve camera location
    camera = mongo.db['cameras'].find_one({'camera' : "camera_2"})['location'] 
    location_array.append(camera)

    # retrieve all officers' location in a list - use google maps api to calculate distance then choose the shortest one
    for i in cursor:
        location_array.append(i['location'])
    
    # call compute distance function here
    compute_distance(location_array, mongo) 