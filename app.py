import cv2 , os, joblib
import pandas as pd, numpy as np
from datetime import date, datetime
from sklearn.neighbors import KNeighborsClassifier
import os
from flask import Flask, request, render_template
from Backend.model import *
from Backend.attendance import *


#defining Flask app:
app=Flask(__name__, template_folder='Frontend')

#@ No of images to be taken for training/detection:
nimgs= 10


if not os.path.isdir('static'):          #main file for storing employee data and trained model 
    os.makedirs('static')

if not os.path.isdir('static/faces'):    #for storing faces of employee
    os.makedirs('static/faces')


################## ROUTING FUNCTIONS #########################

# # Our main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# ## List users page
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# ## Delete functionality
# @app.route('/deleteuser', methods=['GET'])
# def deleteuser():
#     duser = request.args.get('user')
#     deletefolder('static/faces/'+duser)

#     ## if all the face are deleted, delete the trained file...
#     if os.listdir('static/faces/')==[]:
#         os.remove('static/face_recognition_model.pkl')
    
#     try:
#         train_model()
#     except:
#         pass

#     userlist, names, rolls, l = getallusers()
#     return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# import joblib

# # Load the trained model
# def load_model():
#     try:
#         model = joblib.load('static/face_recognition_model.pkl')  # Make sure the model path is correct
#         return model
#     except FileNotFoundError:
#         print("Model file not found.")
#         return None

# # Function to identify the face
# def identify_face(face_features):
#     model = load_model()
#     if model:
#         prediction = model.predict(face_features)  # face_features should be reshaped if necessary
#         return prediction
#     else:
#         return None


# # Our main Face Recognition functionality. 
# # This function will run when we click on Take Attendance Button.
# @app.route('/start', methods=['GET'])
# def start():
#     names, rolls, times, l = extract_attendance()

#     if 'face_recognition_model.pkl' not in os.listdir('static'):
#         return render_template('Frontend/home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

#     ret = True
#     cap = cv2.VideoCapture(0)
#     while ret:
#         ret, frame = cap.read()
#         if len(extract_faces(frame)) > 0:
#             (x, y, w, h) = extract_faces(frame)[0]
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
#             cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
#             face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
#             identified_person = identify_face(face.reshape(1, -1))[0] # type: ignore
#             add_attendance(identified_person)
#             cv2.putText(frame, f'{identified_person}', (x+5, y-5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#         cv2.imshow('Attendance', frame)
#         if cv2.waitKey(1) == 27:
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     names, rolls, times, l = extract_attendance()
#     return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# # A function to add a new user.
# # This function will run when we add a new user.
# @app.route('/add', methods=['GET', 'POST'])
# def add():
#     newusername = request.form['newusername']
#     newuserid = request.form['newuserid']
#     userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
#     if not os.path.isdir(userimagefolder):
#         os.makedirs(userimagefolder)
#     i, j = 0, 0
#     cap = cv2.VideoCapture(0)
#     while 1:
#         _, frame = cap.read()
#         faces = extract_faces(frame)
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
#             cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
#             if j % 5 == 0:
#                 name = newusername+'_'+str(i)+'.jpg'
#                 cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
#                 i += 1
#             j += 1
#         if j == nimgs*5:
#             break
#         cv2.imshow('Adding new User', frame)
#         if cv2.waitKey(1) == 27:
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     print('Training Model')
#     train_model()
#     names, rolls, times, l = extract_attendance()
#     return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# # Our main function which runs the Flask App
# if __name__ == '__main__':
#     app.run(debug=True)

