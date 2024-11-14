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
#cap = cv2.VideoCapture(0)
#ESP32:
#cap = cv2.VideoCapture(ip)

cv2.namedWindow("test")

sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

while True:
    imgResponse = urllib.request.urlopen(ip, timeout=5)
    imgNp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
    frame=cv2.imdecode(imgNp, -1)
    #cv2.imshow("test", img)
    time.sleep(.2)
    #ret, frame = cap.read()

    cv2.imshow("Frame", frame)

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)

    #checking dist
    #face_recognition.compare_faces([], , 0.5)

    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
        
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

# cap.release()
cv2.destroyAllWindows()