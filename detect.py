import cv2
import numpy as np
import sqlite3

# Load the trained model from the recognizer folder
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer/trainingdata.yml')

# Load the Haar cascade for face detection from the specified path
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to get the profile data from the database for a given ID
def get_profile(id):
    conn = sqlite3.connect("database.db")
    query = "SELECT * FROM STUDENT WHERE Id=?"
    cursor = conn.execute(query, (id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

# Start the camera
cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Check if confidence is less than 100 ==> "0" is a perfect match
        if confidence < 100:
            profile = get_profile(id)
            if profile:
                cv2.putText(img, f"{profile[1]}, {profile[2]}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(img, "Unknown", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Face', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()