import face_recognition
import cv2
import os
import glob
import numpy as np
import urllib.request
import urllib.parse



class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.ctr=0

        # Resize frame for a faster speed
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """
        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print("IMIMIMIMIMIMI")
            print(img)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Get encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Store file name and file encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
            print(img_encoding)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        # tolerance = 0.5

        # verified = False
        # for c_face_encoding in face_encodings:
        #     for p_face_encoding in p_encodings:
        #         if(face_recognition.face_distance(p_face_encoding, c_face_encoding) >= tolerance):
        #             #means the data was already sent to server for verification
        #             verified = True
        #             break
        # if(not verified):
        #     #send request to server
        #     print("hi")
        # return face_encodings


        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            print(matches)
            # print("hihi")
            # print(self.known_face_encodings)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = self.known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names

    def request(self, encoding):
        url='http://localhost:3000/check_id'
        data = {
            "embedding": encoding
        }
        data = urllib.parse.urlencode(data).encode('utf-8')
        req = urllib.request.Request(url, data=data)
        req.add_header('Content-Type', 'application/x-www-form-urlencoded')
        with urllib.request.urlopen(req) as response:
            response_data = response.read().decode('utf-8')
        print(response_data)


    def verified(self, frame, p_encodings):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
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
            self.request(face_encodings[0])
        return face_encodings