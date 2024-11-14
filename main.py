import cv2
import urllib.request
import urllib.parse
import numpy as np
from simple_facerec import SimpleFacerec
import os
from dotenv import load_dotenv
import time
import face_recognition
import json

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

frame_resizing = 0.25
request_status = False

#maybe make this async?
def request(encoding):
    global request_status
    if(not request_status):
        request_status=True
        url='http://localhost:3000/check_id'
        data = {
            "embedding": encoding
        }
        #encoding - can't send np array??
        data=json.dumps(data)
        # data = urllib.parse.urlencode(data).encode('utf-8')
        req = urllib.request.Request(url, data=data)
        req.add_header('Content-Type', 'application/json')
        with urllib.request.urlopen(req) as response:
            response_data = response.read().decode('utf-8')
        print(response_data)
        request_status=False


def verified(frame, p_encodings):
    small_frame = cv2.resize(frame, (0, 0), fx=frame_resizing, fy=frame_resizing)
    # Find all the faces and face encodings in the current frame of video
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    tolerance = 0.5

    verified = False
    if((len(p_encodings)>0) and len(face_encodings)>0):
        for c_face_encoding in face_encodings:     
            if(face_recognition.face_distance(p_encodings, c_face_encoding)):
                verified=True
                break
        #print(face_encodings)
    if(not verified and len(face_encodings)>0):
        #send request to server
        #for each face
        # for encoding in face_encodings:
        #     self.request(encoding)

        #for the first face
        request(face_encodings[0])

    return face_encodings

embeddings = []

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
    embeddings = verified(frame, embeddings)

    #checking dist
    #face_recognition.compare_faces([], , 0.5)

    # for face_loc, name in zip(face_locations, face_names):
    #     y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
    #     cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
        
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

# cap.release()
cv2.destroyAllWindows()