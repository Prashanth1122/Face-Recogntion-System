import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = "dataset"  # Corrected the typo in the path name

def get_images_with_id(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]  # Skip hidden files
    faces = []
    ids = []
    for image_path in image_paths:
        face_img = Image.open(image_path).convert('L')  # Corrected variable name and method call
        face_np = np.array(face_img, 'uint8')  # Corrected data type
        id = int(os.path.split(image_path)[-1].split(".")[1])  # Corrected os.split to os.path.split
        print(id)
        faces.append(face_np)
        ids.append(id)
        cv2.imshow("Training", face_np)  # Corrected syntax for imshow
        cv2.waitKey(10)
    return np.array(ids), faces

ids, faces = get_images_with_id(path)  # Corrected function name
recognizer.train(faces, ids)  # Corrected the order of arguments
recognizer.save("recognizer/trainingdata.yml")  # Ensure the 'recognizer' directory exists
cv2.destroyAllWindows()  # Corrected the typo in the function name