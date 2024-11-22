import pandas as pd, os, cv2
from datetime import datetime, date
from pymongo import MongoClient
from datetime import date

# MongoDB Setup
client = MongoClient('mongodb://localhost:27017/')  # Update with your MongoDB URI
db = client['attendance_system']  # Use or create the database
datetoday = date.today().strftime("%m_%d_%y")
attendance_collection_name = f'Attendance-{datetoday}'
attendance_collection = db[attendance_collection_name]

if attendance_collection_name not in db.list_collection_names():
    # Initialize the collection by adding a header document (optional)
    attendance_collection.insert_one({
        "Name": None,  # Placeholder
        "Roll": None,  # Placeholder
        "Time": None   # Placeholder
    })
    # Optionally delete the placeholder later if not needed
    attendance_collection.delete_one({"Name": None, "Roll": None, "Time": None})


#@ Initializing VideoCapture object to access cam:
face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

datetoday=date.today().strftime("%m_%d_%y")
datetoday2=date.today().strftime('%D-%M-%Y')

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


# Extract info from today's attendance file in attendance folder
def extract_attendance():
    # df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')

    # if 'Roll' in df.columns:
    #     rolls=df['Roll']
    # else: 
    #     print('Roll cannot be found')
    #     rolls=[]

    # names = df['Name']
    # rolls = df['Roll']
    # times = df['Time']
    # l = len(df)
    # return names, rolls, times, l

    records = attendance_collection.find()
    names, rolls, times = [], [], []
    for record in records:
        names.append(record['Name'])
        rolls.append(record['Roll'])
        times.append(record['Time'])

    l = len(names)
    return names, rolls, times, l


# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    # df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    # if int(userid) not in list(df['Roll']):
    #     with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
    #         f.write(f'\n{username},{userid},{current_time}')

      # Check if the roll number already exists for today
    if not attendance_collection.find_one({"Roll": int(userid)}):
        # Insert the attendance record
        attendance_collection.insert_one({
            "Name": username,
            "Roll": int(userid),
            "Time": current_time
        })
    


## A function to get names and rol numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


## A function to delete a user folder 
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)
