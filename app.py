import cv2 , os, joblib
import pandas as pd, numpy as np
from datetime import date, datetime


#@ No of images to be taken for training:
nimgs= 10


#@ Saving Date today in 2 different formats:
datetoday=date.today().strftime("%m_%d_%y")
datetoday2=date.today().strftime('%D-%M-%Y')


#@ Creating directories:
if not os.path.isdir('Attendance'):       #for recording attendance
    os.makedirs('Attendance')

if not os.path.isdir('static'):          #main file for storing employee data and trained model 
    os.makedirs('static')

if not os.path.isdir('static/faces'):    #for storing faces of employee
    os.makedirs('static/faces')

if f'Attendance-{datetoday}.json' not in os.listdir('Attendance'):  #for writing/keeping attendance in json format on daily basis on detecting faces  
    with open(f'Attendance/Attendance-{datetoday}.json', 'w') as f:
        f.write('Name, Id, Time')


#@ Getting total numbers of registered users:
def totalreg():
    return len(os.listdir('static/faces'))