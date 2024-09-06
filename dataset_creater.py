import cv2
import numpy as np
import sqlite3

# Load the cascade
try:
    faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    print(f"Error loading cascade classifier: {e}")
    exit()

cam = cv2.VideoCapture(0)

# ... rest of your code ...

# Ensure the STUDENT table exists
create_table_if_not_exists()

Id = input('enter user Id: ')
Name = input('enter user name: ')
age = input('enter user age: ')

insert_or_update(Id, Name, age)

sampleNum = 0
while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        sampleNum += 1
        cv2.imwrite("dataset/user." + str(Id) + "." + str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.waitKey(100)
        
    cv2.imshow("Face", img)
    if cv2.waitKey(1) & 0xFF == ord('q') or sampleNum > 20:
        break
    
cam.release()
cv2.destroyAllWindows()