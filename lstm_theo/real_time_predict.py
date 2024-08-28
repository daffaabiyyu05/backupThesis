#!/usr/bin/env python
# coding: utf-8

import copy
import cv2
import mediapipe as mp

import numpy as np
import time 

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

print("loading Tensorflow....")
import tensorflow as tf

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
    dim_size = 256
    face_image = np.zeros((h,w,c), np.uint8) #create a blank image copy
    final_image = np.zeros((dim_size,dim_size,3), np.uint8) #create a blank cropped image copy
    landmark_position = np.zeros([468,3]) #empty landmark position
    data_norm = np.zeros([468,2])

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
              image=face_image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_TESSELATION,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_tesselation_style())
            
            for i in range(468):
                landmark_position[i,0]=face_landmarks.landmark[i].x
                landmark_position[i,1]=face_landmarks.landmark[i].y
                landmark_position[i,2]=face_landmarks.landmark[i].z
            
            final_image = face_image
            data = landmark_position
            data = np.delete(data, 2, 1)

            #START NORM
            x_arr = data[:,0]
            y_arr = data[:,1]

            x_norm = (x_arr - x_arr.min())/(x_arr.max() - x_arr.min())
            y_norm = (y_arr - y_arr.min())/(y_arr.max() - y_arr.min())

            data_norm [:,0] = x_norm
            data_norm [:,1] = y_norm
            #END NORM

            #CV2 View
            w = 800
            h = 800
            view = np.zeros((w, h))
            for i in range(468):
                xp=(int) (x_norm[i]*w)
                yp=(int) (y_norm[i]*h)
                cv2.circle(view, (xp,yp), radius=1, color=(255, 255, 255), thickness=1)
            cv2.imshow('Landmark', view)

    return final_image, data_norm

print("Loading model...")
model_paths = ['./best_model_cnn_lstm_norm_lm_xy_conv1d4u5k_dense4_lstm8_8_100_3', './best_model_cnn_bilstm_norm_lm_xy_conv1d4u5k_dense4_bilstm8_8_100_3']
model_path = model_paths[1] +'.h5'
print(model_path)
model_tf_lstm = tf.keras.models.load_model(model_path)
model_classes = ['A', 'E', 'I', 'O', 'U','-']

        
TimeStart = time.time() 
TimeNow = time.time() 
FrameRate = 30
videoMode = False
trackLandmark = True

if videoMode:
    ### For video input:
    cap = cv2.VideoCapture('./') #insert video path
else:
    ### For webcam input:
    cap = cv2.VideoCapture(0)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9
)

stored_frames = []
BATCH_SIZE = 1
TARGET_FRAME = 30
FRAME_GAP = 1
font = cv2.FONT_HERSHEY_SIMPLEX
predict = "*"
second_predict = "*"
double_prediction = False
print("Starting...")
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        if videoMode:
            break
        else:
            continue

    image = image
    faceImg, data = ExtrakLandmark(face_mesh,image)
  
    TimeNow = time.time() 
    if TimeNow - TimeStart > 1 / FrameRate:
        TimeStart=TimeNow

        stored_frames.append(data)
        
        if len(stored_frames) > TARGET_FRAME:
            for i in range(FRAME_GAP):
                del stored_frames[0]
        
        if len(stored_frames) == TARGET_FRAME:
            print("start predict")
            images_array = np.array(stored_frames)
            print(images_array.shape)
            images_array = np.reshape(images_array, (BATCH_SIZE, TARGET_FRAME, 468, 2))
            print(images_array.shape)
            print("predicting model")
            model_results = model_tf_lstm.predict(images_array, verbose = 0)[0]
            print("model predicted")
            merged_prediction = list(zip(model_classes, model_results))
            
            print(merged_prediction)
            
            merged_prediction.sort(key=lambda a: a[1])
            
            print("Prediction :", merged_prediction[-1])
            print(" ")
            predict = str(merged_prediction[-1][0])
            second_predict = str(merged_prediction[-2][0])

        # Flip the image horizontally for a selfie-view display.
        displayImg = cv2.flip(image, 1)
        meshImg = cv2.flip(faceImg, 1)
        cv2.putText(displayImg, 
                str(predict), 
                (50, 50), 
                font, 1, 
                (255, 255, 255), 
                2, 
                cv2.LINE_4)
        if double_prediction:
            cv2.putText(displayImg, 
                str(second_predict), 
                (50, 100), 
                font, 1, 
                (255, 255, 255), 
                2, 
                cv2.LINE_4)
        cv2.imshow('Display', displayImg)
        #cv2.imshow('Mesh', meshImg)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

