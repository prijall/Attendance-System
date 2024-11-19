import cv2 , os, joblib
import pandas as pd, numpy as np
from datetime import date, datetime
from sklearn.neighbors import KNeighborsClassifier


#@ No of images to be taken for training/detection:
nimgs= 10


#@ Saving Date today in 2 different formats:
datetoday=date.today().strftime("%m_%d_%y")
datetoday2=date.today().strftime('%D-%M-%Y')


#@ Initializing VideoCapture object to access cam:
face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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


#@ Extracting the face from an image:
def extract_faces(img):
    try:
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points=face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20)) #pyramid technique
        return face_points
    except:
        return []


#@ Identifying face using ML Model:
def identify_face(facearray): # face array will have characteristics of a face.
    model=joblib.load('static/face_recognition_model.pkl') #loading pre-trained model
    return model.predict(facearray)

#@ for training model on all the faces available in faces folder:
faces=[] #stores flattened arrays of image pixel values
labels=[] # stores user names for each image
userlist=os.listdir('static/faces')
for user in userlist:
    for imgname in os.listdir(f'static/faces/{user}'): #all the images in folder
        img=cv2.imread(f'static/faces/{user}/{imgname}')
        resized_face=cv2.resize(img, (50, 50)) # ensuring 50 x 50 pixels
        faces.append(resized_face.ravel()) # converting 50 x 50 x 3 RGB image into 1D array of 7500 values
        labels.append(user)


faces=np.array(faces) #converting in numpy array

#@ Using KNN for training:
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(faces, labels)
joblib.dump(knn, 'static/face_recognition_model.pkl')
