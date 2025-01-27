import torch
import cv2
import pymongo
from datetime import datetime
from PIL import Image
from torchvision import transforms
import os
from Model_architecture import FaceRecognitionCNN  # Import your model architecture

# MongoDB Setup
client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['attendance_db']
collection = db['attendance_log']

# Load the trained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FaceRecognitionCNN(num_classes=len(os.listdir(os.path.join('Dataset', 'train')))).to(device)
model.load_state_dict(torch.load('face_recognition_model.pth', map_location=device, weights_only=True))  # Ensure model is loaded correctly
model.eval()

# Define transformations for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Initialize the face detector (Haar Cascade)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# MongoDB function to log attendance
def log_attendance(user_name):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Check if this user has already logged in today to avoid duplicate attendance
    existing_record = collection.find_one({"name": user_name, "timestamp": {"$gte": timestamp.split(' ')[0]}})
    if not existing_record:
        record = {"name": user_name, "timestamp": timestamp}
        collection.insert_one(record)

# Prediction function using the CNN model
def predict_face(face):
    face = transform(face).unsqueeze(0).to(device)  # Add batch dimension and move to device (GPU/CPU)
    with torch.no_grad():
        output = model(face)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# Real-time face detection and attendance logging

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the face region from the frame
        face = frame[y:y + h, x:x + w]
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))  # Convert to PIL image
        
        # Predict the user based on the CNN model
        label = predict_face(face_pil)
        user_name = os.listdir(os.path.join('Dataset', 'train'))[label]  # Mapping label to user name
        
        # Log attendance in MongoDB for each detected face
        log_attendance(user_name)
        
        # Display the user name on the frame above the bounding box
        cv2.putText(frame, user_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the video stream with bounding boxes and names for multiple faces
    cv2.imshow("Face Recognition - Attendance System", frame)

    # Exit the loop if the user presses 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
