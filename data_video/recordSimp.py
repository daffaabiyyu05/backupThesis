# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:02:39 2023

@author: daffa
"""

import copy
import cv2
import mediapipe as mp
import numpy as np
import os
import datetime
import time 
  

def GetFileName():
        x = datetime.datetime.now()
        s = x.strftime('%Y-%m-%d-%H%M%S%f')
        return s
    
def CreateDir(path):
    ls = [];
    head_tail = os.path.split(path)
    ls.append(path)
    while len(head_tail[1])>0:
        head_tail = os.path.split(path)
        path = head_tail[0]
        ls.append(path)
        head_tail = os.path.split(path)   
    for i in range(len(ls)-2,-1,-1):
        print(ls[i])
        sf =ls[i]
        isExist = os.path.exists(sf)
        if not isExist:
            os.makedirs(sf)

def ExtrakLandmark(face,img):
    br,kl ,w= img.shape
    image =copy.copy(img) 
    #image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face.process(image)

    # Draw the face annotations on the image.
    #image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    (h, w, c) = img.shape[:3]
    face_image = np.zeros((h,w,c), np.uint8) #create a blank image copy

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
              image=face_image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_TESSELATION,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_tesselation_style())

    return face_image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

#RECORD SETTINGS
recordOn = False
modeBW = False
TimedRecord = True
FrameLimit = 300

TimeStart = time.time() 
TimeNow = time.time() 
FrameRate = 30
Counter = 0
recordCount = 0
FrameLeft = FrameLimit

BasePath = "D:\\Thesis\\data_video\\DatasetB\\"
sNamaDirektori = GetFileName() 
sDirektoriData = BasePath + sNamaDirektori

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=[255,255,255])
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
original = cv2.VideoWriter(sDirektoriData + ".avi", fourcc, FrameRate, (640,  480))
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    faceImg = ExtrakLandmark(face_mesh,image)
  
    TimeNow = time.time() 
    if TimeNow - TimeStart > 1/FrameRate and recordOn:
        original.write(image)
        if TimedRecord:
            FrameLeft = FrameLeft - 1
            if FrameLeft <= 0:
                recordOn = not recordOn
                FrameLeft = FrameLimit
                original.release()
                sNamaDirektori = GetFileName() 
                sDirektoriData = BasePath + sNamaDirektori
                original = cv2.VideoWriter(sDirektoriData + ".avi", fourcc, FrameRate, (640,  480))
                recordCount = recordCount + 1
        TimeStart=TimeNow
    
    #switch image based on Mode
    if modeBW:
        activeView = faceImg
    else:
        activeView = image
    #font for record state
    font = cv2.FONT_HERSHEY_SIMPLEX
    activeView = cv2.flip(activeView, 1)
    
    if TimedRecord:
            cv2.putText(activeView, 
                'Timed Mode', 
                (50, 150), 
                font, 1, 
                (0, 255, 0), 
                2, 
                cv2.LINE_4)

    if recordOn:
        cv2.putText(activeView, 
                'RECORDING', 
                (50, 50), 
                font, 1, 
                (0, 0, 255), 
                2, 
                cv2.LINE_4)
        if TimedRecord:
            cv2.putText(activeView, 
                str(FrameLeft), 
                (50, 100), 
                font, 1, 
                (0, 0, 255), 
                2, 
                cv2.LINE_4)
    else:
        cv2.putText(activeView, 
                'NOT RECORDING', 
                (50, 50), 
                font, 1, 
                (255, 255, 255), 
                2, 
                cv2.LINE_4)
        cv2.putText(activeView, 
                str(recordCount), 
                (50, 100), 
                font, 1, 
                (255, 255, 255), 
                2, 
                cv2.LINE_4)
        
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', activeView)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('w') or key == ord('W'):
        modeBW = not modeBW
    if key == ord('r') or key == ord('R'):
        recordOn = not recordOn
        if (not recordOn):
            original.release()
            sNamaDirektori = GetFileName() 
            sDirektoriData = BasePath + sNamaDirektori
            original = cv2.VideoWriter(sDirektoriData + ".avi", fourcc, FrameRate, (640,  480))
            recordCount = recordCount + 1
    if key == ord('t') or key == ord('T'):
        TimedRecord = not TimedRecord
    if key == 27:
        break
cap.release()
original.release()
cv2.destroyAllWindows()