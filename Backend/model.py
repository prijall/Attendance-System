import cv2, os, joblib, numpy as np
from sklearn.neighbors import KNeighborsClassifier

#@ for training model on all the faces available in faces folder:
def train_model():
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
    joblib.dump(knn, 'static/face_recognition_model.pkl') #saving the trained model

    