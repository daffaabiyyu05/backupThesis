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
from tqdm import tqdm
from glob import glob
  

def GetFileName():
        x = datetime.datetime.now()
        s = x.strftime('%Y-%m-%d-%H%M%S%f')
        return s
    
def CreateDir(path):
    ls = []
    head_tail = os.path.split(path)
    ls.append(path)
    while len(head_tail[1])>0:
        head_tail = os.path.split(path)
        path = head_tail[0]
        ls.append(path)
        head_tail = os.path.split(path)   
    for i in range(len(ls)-2,-1,-1):
        #print(ls[i])
        sf =ls[i]
        isExist = os.path.exists(sf)
        if not isExist:
            os.makedirs(sf)

def GenerateDirectories(targetDirectory):
    CreateDir(targetDirectory)
    CreateDir(targetDirectory+"\\original")
    CreateDir(targetDirectory+"\\face-mp")
    CreateDir(targetDirectory+"\\pos-landmark")
    CreateDir(targetDirectory+"\\pos-pixel")
    CreateDir(targetDirectory+"\\face-mp-crop")
    CreateDir(targetDirectory+"\\face-mp-crop-64")
    CreateDir(targetDirectory+"\\face-mp-crop-128")
    CreateDir(targetDirectory+"\\face-mp-crop-256")
    CreateDir(targetDirectory+"\\face-mp-crop-512")
    CreateDir(targetDirectory+"\\face-mp-crop-1024")
    CreateDir(targetDirectory+"\\face-mp-append-300")
    print("Directories created for: " + targetDirectory)

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
    cropped_image = np.zeros((1,1,1), np.uint8) #create a blank cropped image copy
    landmark_position = np.zeros([468,3]) #empty landmark position
    pixel_position = np.zeros([468,2]) #empty landmark position in pixel form --> for cropping purposes

    ctolerance = 5 #margin tolerance to avoid node at edge

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

            for i in range(468):
                pixel_position[i,0]=(int) (face_landmarks.landmark[i].x*w)
                pixel_position[i,1]=(int) (face_landmarks.landmark[i].y*h)
            
            min = np.nanmin(pixel_position, axis = 0 ).astype(int)
            min[min < ctolerance] = ctolerance
            max = np.nanmax(pixel_position, axis = 0 ).astype(int)
            if (max[0] > w):
                max[0] = w
            if (max[1] > h):
                max[1] = h

            gap = [max[0] - min[0], max[1] - min [1]]
            maxgap = np.nanmax(gap, axis = 0).astype(int)
            gap = [int((maxgap - gap[0])/2) + ctolerance , int((maxgap - gap[1])/2) + ctolerance]

            if gap[0] > min[0]:
                gap[0] = min[0]

            if gap[1] > min[1]:
                gap[1] = min[1]

            cropped_image = face_image.copy()
            cropped_image = cropped_image[min[1]-gap[1] : max[1]+gap[1], min[0]-gap[0] : max[0]+gap[0]]

    return face_image, landmark_position, pixel_position, cropped_image

def dataGenerator(ctr, img, fcImg, lmPos, pxPos, crImg, path):
    counterStr = str(ctr).rjust(6, '0')

    #Original Image
    sfc = path+"\\original\\"+ counterStr
    filename = sfc+".jpg"
    cv2.imwrite(filename, img)

    #FaceMesh Image
    sfc = path+"\\face-mp\\"+ counterStr + "-face"
    filename = sfc+".png"
    cv2.imwrite(filename, fcImg)

    #Landmark Position
    sfc = path+"\\pos-landmark\\"+ counterStr + "-lm"
    np.savetxt(sfc+".csv", lmPos, delimiter=',')

    #Pixel Position
    sfc = path+"\\pos-pixel\\"+ counterStr + "-px"
    np.savetxt(sfc+".csv", pxPos, delimiter=',', fmt='%d')

    #Cropped FaceMesh Image
    sfc = path+"\\face-mp-crop\\"+ counterStr + "-facecrop"
    filename = sfc+".png"
    cv2.imwrite(filename, crImg)

    #Scaled Cropped FaceMesh Image
    sfc = path+"\\face-mp-crop-"
    cdim = [64, 128, 256, 512, 1024]
    fname = ""
    h,w,c = crImg.shape[:3]
    for dim_size in cdim:
        if h < dim_size:
            interpolationType = cv2.INTER_LANCZOS4
        else:
            interpolationType = cv2.INTER_AREA
        
        scaledCropImg = cv2.resize(crImg, (dim_size, dim_size), interpolation = interpolationType)
        fname = sfc + str(dim_size) + "\\" + counterStr + ".png"
        cv2.imwrite(fname, scaledCropImg)

    #Cropped-Appended FaceMesh Image
    sfc = path+"\\face-mp-append-300\\"+ counterStr
    filename = sfc+".png"
    append_h = 300
    append_w = 300
    if append_h >= h and append_w >= w:
        append_image = np.zeros((append_h,append_w,c), np.uint8) #create a blank image copy
        append_image[append_h-h:append_h, append_w-w:append_w] = crImg
        cv2.imwrite(filename, append_image)

def parseVideo(capRecordFile, destDirectory="", destDirectories = [], viewImage = False, fileDirectories=[], train_max_frame = 99999, train_max_low_frame = 99999):
    if capRecordFile == 0:
        print("Video not found")
        return

    cap = cv2.VideoCapture(capRecordFile)
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    modeBW = False
    
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total = frame_total)
    Counter = 0

    frame_per_second = int(cap.get(cv2.CAP_PROP_FPS))
    #print("Frame per Second: " + str(frame_per_second))
    doubleFrame = False
    if frame_per_second > 50:
        doubleFrame = True
        nameSecondDir = GetFileName()
        if destDirectory == "":
            destDirectory = destDirectories[1]
        if destDirectory == "":
            print("ERROR: Destination Directory not specified for double frame")
            return
        doubleFrameDirectory = destDirectory + nameSecondDir
        GenerateDirectories(doubleFrameDirectory)
        print("Double FPS: Second Directory Generated")

    max_frame_training = train_max_frame
    if frame_total < train_max_frame or (doubleFrame and frame_total/2 < train_max_frame):
        max_frame_training = train_max_low_frame
    
    if max_frame_training < train_max_low_frame:
        max_frame_training = frame_per_second

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            progress_bar.close()
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break
        
        faceImg, landmarkPos, pixelPos, cropImg = ExtrakLandmark(face_mesh,image)
        Counter = Counter + 1
        if (Counter <= max_frame_training) or (Counter <= (max_frame_training*2) and doubleFrame):
            directoryData = fileDirectories[0] #train
        else:
            directoryData = fileDirectories[1] #test

        if not doubleFrame: #singular frame
            dataGenerator(Counter, image, faceImg, landmarkPos, pixelPos, cropImg, directoryData)
        else:
            if Counter%2 == 0: #even frames
                dataGenerator(Counter/2, image, faceImg, landmarkPos, pixelPos, cropImg, directoryData)
            else: #odd frames
                dataGenerator((Counter+1)/2, image, faceImg, landmarkPos, pixelPos, cropImg, doubleFrameDirectory)
        
        #switch image based on Mode
        if modeBW:
            activeView = faceImg
        else:
            activeView = image

        activeView = cv2.flip(activeView, 1)
            
        # Flip the image horizontally for a selfie-view display.
        if viewImage:
            cv2.imshow('MediaPipe Face Mesh', activeView)
        
        progress_bar.update(1)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('w') or key == ord('W'):
            modeBW = not modeBW
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

BasePath = "D:\\Thesis\\data_n3\\"

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=[255,255,255])

#NODE INFORMATION --> UPDATE AS NECESSARY
classes = ['-', 'A', 'E', 'I', 'O', 'U']
#class processing
start_class_processing = 0
max_class_processed = 6
#nodes
first_node = [15, 19, 15, 15, 14, 5]
last_node = [15, 19, 15, 15, 14, 14]
    
TrainPath = BasePath + "train\\"
TestPath = BasePath + "test\\"
    

start = time.time()
class_index = 0

for TargetLetter in classes:
    class_index = class_index + 1
    print("Current Class: " + TargetLetter)
    if class_index < start_class_processing:
        print("Below the minimum specified class, skipping this class")
        continue

    if class_index > max_class_processed:
        print("Over the class limit, aborting process")
        break

    start_letter = time.time()
    image_path = "./" + TargetLetter + '/'
    base_letter_paths = [TrainPath + TargetLetter + "\\", TestPath + TargetLetter + "\\"]
    enforce_data_480p = False

    print("Fetching Video")
    all_images = []
    counter = 0
    for path in sorted(glob(image_path + "*.*")):
        counter += 1
        if counter <= first_node[class_index-1] or counter > last_node[class_index-1] or (path.endswith(".mp4") and enforce_data_480p):
            print("Skipping file: " + path)
            continue

        if path.endswith(".mp4"):
                print("WARNING: Data from this video might not be compatible with the appending method")

        sNamaDirektori = GetFileName()
        fileDirectoryList = []
        sDirektoriData = base_letter_paths[0] + sNamaDirektori 
        GenerateDirectories(sDirektoriData)
        fileDirectoryList.append(sDirektoriData)
        sDirektoriData = base_letter_paths[1] + sNamaDirektori 
        GenerateDirectories(sDirektoriData)
        fileDirectoryList.append(sDirektoriData)
        print("Processing video: " + path)
        parseVideo(path, destDirectories=base_letter_paths,fileDirectories=fileDirectoryList, train_max_frame=600, train_max_low_frame=150)

    end_letter = time.time()
    print(str(end_letter - start_letter) + " seconds to finish process for letter " + TargetLetter)
    local_time = time.ctime(end_letter)
    print("Local time:", local_time)

end = time.time()

local_time = time.ctime(end)
print("Local time:", local_time)

print(str(end - start) + " seconds to finish entire process.")