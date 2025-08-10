This project is a real-time facial recognition attendance system built using Python, OpenCV, and the face_recognition library. It identifies registered individuals through webcam input and logs their attendance with timestamps in a CSV file.

Features
Detects and recognizes faces from a live webcam feed

Compares detected faces with stored images in a specified folder

Automatically records attendance with a timestamp

Stores attendance logs in Attendance.csv

Requirements
Python 3.x

OpenCV

face_recognition

NumPy

pandas

Install dependencies:

pip install opencv-python face_recognition numpy pandas
How It Works
Load known face images from the images directory.

Encode faces and store their data.

Use webcam to capture frames in real-time.

Compare detected faces to stored encodings.

Log recognized names and timestamps to Attendance.csv.

Running the Project
Clone the repository.

Create an images folder and add labeled face images (file names as person names).

Run the script:

bash
Copy
Edit
python main.py
Press q to quit the program.

