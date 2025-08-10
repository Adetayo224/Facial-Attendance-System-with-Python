import pandas as pd
import cv2
import numpy as np
import os
from datetime import datetime
import face_recognition

# Define the path for the images and attendance directory
path = r'C:\Users\adeta\Desktop\Facial Attendance System\images'
attendance_dir = os.path.join(os.getcwd(), 'attendance')
attendance_file_path = os.path.join(attendance_dir, 'Attendance.csv')

# Ensure the 'attendance' directory exists
if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)

# Check if the Attendance.csv file exists in the 'attendance' directory
if os.path.exists(attendance_file_path):
    print("Attendance.csv already exists..")
    os.remove(attendance_file_path)  # Remove existing attendance file
else:
    df = pd.DataFrame(list())  # Create a new empty DataFrame
    df.to_csv(attendance_file_path)  # Save it as Attendance.csv

# Load images and extract class names
images = []
classNames = []
myList = os.listdir(path)  # List all image files in the directory
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')  # Read image
    images.append(curImg)  # Append to image list
    classNames.append(os.path.splitext(cl)[0])  # Extract and store class name
print(classNames)

# Function to find encodings for the loaded images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        encode = face_recognition.face_encodings(img)[0]  # Encode the face
        encodeList.append(encode)
    return encodeList

# Function to mark attendance by writing name and timestamp in the CSV
def markAttendance(name):
    with open(attendance_file_path, 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]  # Extract names

        if name not in nameList:  # Check if the name is already recorded
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')  # Current time
            f.writelines(f'\n{name},{dtString}')  # Write new entry

# Find encodings for all loaded images
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()  # Capture frame from webcam
    if not success:
        print("Failed to capture image")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize frame for faster processing
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Find all faces and encodings in the current frame
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        # Compare the detected face with known encodings
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)  # Get the index of the closest match

        if matches[matchIndex]:  # If a match is found
            name = classNames[matchIndex].upper()  # Get the matched name
        else:
            name = "UNKNOWN"  # Set name as "UNKNOWN" for unrecognized faces

        # Scale the face location back to original frame size
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

        # Draw a rectangle around the face
        color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)  # Green for recognized, Red for unknown
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)

        # Display the name below the face rectangle
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Mark attendance only if the face is recognized
        if name != "UNKNOWN":
            markAttendance(name)

    # Display the webcam feed with face recognition
    cv2.imshow('Webcam', img)

    # Break the loop if 'q' is pressed
    key = cv2.waitKey(5)
    if key == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
