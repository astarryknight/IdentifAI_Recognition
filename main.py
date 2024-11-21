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
from collections import Counter

load_dotenv() 

ip = os.getenv("IP")
print(ip)

a=np.array([
-0.12542816, 0.0652061, 0.12394607, 0.02423882, -0.08168113, 0.01228087, -0.01443338, -0.11215433, 0.19984981, -0.07825743, 0.16019039, 0.02497483, -0.1638823, 0.08152732, -0.07165051, 0.04151121, -0.12665054, -0.08681241, -0.02023332, -0.12555268, 0.03211293, 0.03364171, -0.0618048, -0.00107205, -0.20101814, -0.27922413, -0.08259065, -0.10732785, 0.10351838, -0.05197021, -0.02078326, 0.03370402, -0.14583838, 0.02219997, 0.04582398, 0.03431956, 0.02884571, 0.01754401, 0.18889967, -0.00322025, -0.1391664, 0.03618713, 0.0616816, 0.21097834, 0.16537064, 0.13099001, -0.00756426, -0.10378352, 0.09754878, -0.17788398, 0.03757026, 0.13508025, 0.06225245, 0.06666714, 0.0413729, -0.22560169, -0.03160803, -0.02334791, -0.1824531, 0.12866431, 0.06347475, -0.10380627, -0.05282435, 0.00439168, 0.20403142, 0.09930803, -0.1003943, -0.11790477, 0.1848211, -0.24838838, -0.01500291, 0.04746493, -0.06079706, -0.14784619, -0.19681098, 0.0648348, 0.48041481, 0.17566882, -0.15714394, 0.03415987, -0.03032983, -0.04741704, 0.09904169, 0.09931719, -0.13309522, -0.01780549, 0.00637066, 0.03630227, 0.23629537, 0.07084066, -0.01243448, 0.18567468, -0.01211701, 0.0697848, -0.01082286, 0.01339467, -0.12580273, -0.02445328, -0.06693625, -0.0406842, 0.03563322, -0.01847049, 0.09503315, 0.15868054, -0.22116125, 0.18701367, -0.0545835, -0.06322001, 0.00734232, 0.08494901, -0.06663164, 0.01104206, 0.18594857, -0.2523855, 0.22635087, 0.15666212, 0.04192548, 0.14160854, 0.03779654, 0.04590216, 0.05449094, 0.05995531, -0.14256194, -0.08043612, -0.01090786, -0.17400707, 0.09284517, 0.01759548])

#print(np.array([0,1,2,3,4]))

# print("hi")
# print([a][0])
#NEED TO ACCESS LIKE THIS, other wise its fine
#STORE DATA AS A AN ARRAY, CAN BE CONVERTED TO NPARRAY PRETTY EASILY - np.array(array) etc

db = []
embeddings = []

current_faces=[]

def update():
    global db
    global embeddings
    #url='http://localhost:3000/get_faces'
    url='http://127.0.0.1:5000/get_faces'
    req = urllib.request.Request(url)
    content=""
    with urllib.request.urlopen(req) as response:
        content = json.load(response)

    l=len(content)
    c=0
    while(c<l):
        db.append(json.loads(content[c]))
        embeddings.append(np.array(json.loads(content[c])["embeddings"])) #UNCOMMENT FOR PROD PLEASE
        c=c+1
    print(embeddings)
    # print(content)

update() #UNCOMMENT FOR PROD PLEASE

#webcam:
cap = cv2.VideoCapture(0)
#ESP32:
#cap = cv2.VideoCapture(ip)

cv2.namedWindow("test")

# sfr = SimpleFacerec()
# print("MINE")
# for i in embeddings:
#     sfr.server_images(i[0])
# print("THEIRS")
# sfr.load_encoding_images("images/")
# time.sleep(5)

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
        #print(response_data)
        request_status=False

def clean_input(names):
    c_clean = 20 #number of faces to define N
    r_ratio = 0.7 #ratio of faces in past N (c_clean) that need to match for a definite recognition

    # if(len(names)>=c_clean):
    #     a=names.count(db[i]["name"])
    #     r=a/len(names)
    #     if(r>=r_ratio):
    #         return db[i]["id"]
    #     current_faces.pop()
    # current_faces.append(db[i]["name"])
    


def verified(frame, p_encodings):
    global current_faces
    small_frame = cv2.resize(frame, (0, 0), fx=frame_resizing, fy=frame_resizing)
    # Find all the faces and face encodings in the current frame of video
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    tolerance = 0.0005
    c_clean = 100 #number of faces to define N
    r_ratio = 0.7 #ratio of faces in past N (c_clean) that need to match for a definite recognition

    # if((len(p_encodings)>0) and len(face_encodings)>0):
    #     for c_face_encoding in face_encodings:    
    #         print(c_face_encoding, flush=True) 
    #         if(face_recognition.face_distance(p_encodings, c_face_encoding)):
    #             verified=True
    #             break
        #print(face_encodings)
    if(len(face_encodings)>0):
        # print("i see u")
        #send request to server
        #for each face
        # for encoding in face_encodings:
        #     self.request(encoding)

        #for the first face
        #request(face_encodings[0])
        i=0
        # print(len(embeddings))
        # print(len(db))
        while(i<(len(embeddings)-0)):
            emb=embeddings[i] #CHECK THIS LATER
            #print("why tho", flush=True)
            #print([emb])
            matches = face_recognition.compare_faces([emb], face_encodings[0], tolerance)
            #print(type(matches))
            #print(np.isin(matches[0], ))
            #np.any(matches[0][:,0]==np.True_)
            #print(matches.count(True))
            #print(matches[0])
            #print(matches, flush=True)
            if np.any(matches[0]==np.True_):
                # print("i recognize you...")
                print("hi "+str(db[i]["name"]))
            else:
                print("i dont see anyone i know :(")
            # if(face_recognition.face_distance(emb, face_encodings[0]) < tolerance):
            #     print("I recognize you, "+db[i]["name"])
            if(len(current_faces)>=c_clean):
                a=current_faces.count(db[i]["name"])
                r=a/len(current_faces)
                if(r>=r_ratio):
                    print("I KNOW ITS YOU"+db[i]["name"])
                current_faces.pop()
            current_faces.append(db[i]["name"])

            i=i+1
    else:
        print("i dont see anyone i know :(")
    return face_encodings


running_names=[]
def detect_known_faces(frame,encoding_list):
        global running_names
        frame_resizing=0.25
        small_frame = cv2.resize(frame, (0, 0), fx=frame_resizing, fy=frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        id = None
        c_clean = 5 #number of faces to define N
        r_ratio = 0.7 #ratio of faces in past N (c_clean) that need to match for a definite recognition
        # tolerance = 0.5
        
        name = "Unknown"

        face_names = []
        face_dists = []
        for face_encoding in face_encodings:
            i=0#test, but prolly good - bugged out with 2 faces
            for encodings in encoding_list:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(encodings, face_encoding, 0.5)
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(encodings, face_encoding)
                face_dists.append(np.min(face_distances))
                print(face_dists)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = db[i]["name"]#doesnt work for 2 faces?
                    id = db[i]["id"]
                
                face_names.append(name)
                i=i+1

        if(len(face_dists)>0):
            i=np.argmin(face_dists)
            print(face_names)
            name=face_names[i]
            print("TRY THINS")
            print(name)

        #cleaning?
        # if(name != "Unknown"):
        #     if(len(running_names)>=c_clean):
        #         a=running_names.count(name)
        #         r=a/len(running_names)
        #         if(r>=r_ratio):
        #             print("Hi, "+name+". Your ID Number is "+id+".")
        #         running_names.pop()
        #     running_names.append(name)


        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / frame_resizing
        return face_locations.astype(int), face_names, id

# embeddings = []

names_running = []

while True:
    # imgResponse = urllib.request.urlopen(ip, timeout=15)
    # imgNp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
    # frame=cv2.imdecode(imgNp, -1)
    #print('hello?')

    ret, frame = cap.read()

    # cv2.imshow("Frame", frame)

    #Detect Faces
    #face_locations, face_names = sfr.detect_known_faces(frame) #CLEAN INPUTS IN THE DETECTION FUNCTION, SINCE YOU HAVE ACCESS TO INDEX
                                                                #IMPLEMENT FOR MULTIPLE FACES PER PERSON?
    # for i in embeddings:
    #     face_locations, face_names, id = detect_known_faces(frame, i)
    #     if(id!=None):
    #         break

    face_locations, face_names, id = detect_known_faces(frame, embeddings)
    #embeddings = verified(frame, embeddings)

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