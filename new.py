import cv2
import urllib.request
import numpy as np
from simple_facerec import SimpleFacerec
import os
from dotenv import load_dotenv
import time
import face_recognition

load_dotenv() 

ip = os.getenv("IP")
print(ip)

#webcam:
cap = cv2.VideoCapture(0)
#ESP32:
#cap = cv2.VideoCapture(ip)

cv2.namedWindow("test")

sfr = SimpleFacerec()
#sfr.load_encoding_images("images/")

embeddings = []

while True:
    time.sleep(.2)
    ret, frame = cap.read()

    cv2.imshow("Frame", frame)

    # Detect Faces
    embeddings = sfr.verified(frame, embeddings)
        
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

# cap.release()
cv2.destroyAllWindows()