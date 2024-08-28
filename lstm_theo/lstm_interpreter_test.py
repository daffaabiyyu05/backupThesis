#!/usr/bin/env python
# coding: utf-8

import copy
import cv2
import mediapipe as mp

import numpy as np
import time
from tqdm import tqdm
from glob import glob
import gc

print("loading Tensorflow....")
import tensorflow as tf

from tensorflow import keras as K

test_mode = True
print("TEST MODE: " + str(test_mode))
model_paths = ['best_model_cnn_bilstm_norm_lm_xy_conv1d4u5k_dense4_bilstm8_8_100_3'
               ]
base_path = './'
result_suffix = '_norm'
model_count = 0

model_classes = ['A', 'E', 'I', 'O', 'U','-']
checked_letters = model_classes
DatasetCode = '_landmark_norm'
#image_ext = "jpg"
image_ext = "csv"

#Image Load and View Configuration
DEFAULT_SKIP_FRONT = 0
DEFAULT_SKIP_REAR = 999999
SIZE = 468
CHANNEL = 2

skip_image = DEFAULT_SKIP_FRONT #set to DEFAULT_SKIP_FRONT when all image need to be loaded
skip_rear_image = DEFAULT_SKIP_REAR #set to DEFAULT_SKIP_REAR when not used
verbose_mode = 0 #0 for tqdm mode, 1 for print verbose, 2 for print + index verbose; only relevant to the detection section
view_prediction_list = False #Set whether prediction and indexed list is active or not

#Load Configuration:
BATCH_SIZE = 1
TARGET_FRAME = 30
FRAME_GAP = 3

for model_path in model_paths:
    model_count = model_count + 1
    print("Model " + str(model_count) + " of " + str(len(model_paths)))
    print("Loading model...")
    #model_path = 'best_model_lstm_sliced256_2_3'     
    model_tf_lstm = tf.keras.models.load_model(base_path + model_path + '.h5')
    print("Loaded:" + model_path)
    final_result = []
    final_letter_mapping = []
    total_predict = 0
    total_error = 0

    if test_mode:
        base_image_path = './test' + DatasetCode + '/'
        interpreter_result_path = base_path + 'interpreter_test/'
    else:
        base_image_path = './train'+ DatasetCode +'/'
        interpreter_result_path = base_path + 'interpreter_validate/'

    print("Image Base Path:" + base_image_path)
    interpreter_config_path = " G " + str(FRAME_GAP) + " DATA LOAD " + str(checked_letters) + " CLASS ORDER " + str(model_classes)

    print("Map the following letters:"+ str(checked_letters))
    for TargetLetter in checked_letters:
        print("Mapping Prediction Letters for " + TargetLetter)
        predictionCounter = 0
        letter_mapping = [0, 0, 0, 0, 0, 0]

        image_path = base_image_path + TargetLetter + '/'

        print("Fetching Images")
        all_images = []
        counter = 0
        for path in tqdm(sorted(glob(image_path + "*." + image_ext))):
            counter += 1
            if counter < skip_image:
                continue
            if counter > skip_rear_image:
                continue
            image = np.genfromtxt(path, delimiter=',')
            all_images += [image]
        
        print(len(all_images))
        stored_frames = []
        predictionlist = []
        indexedlist = []
        print("Starting...")
        if verbose_mode == 0:
            all_images = tqdm(all_images)

        for idx, image in enumerate(all_images):
            npimg = image
            stored_frames.append(npimg)
            
            if len(stored_frames) > TARGET_FRAME:
                for i in range(FRAME_GAP):
                    del stored_frames[0]
            
            if len(stored_frames) == TARGET_FRAME:
                predictionCounter =+ 1
                if verbose_mode > 1:
                    print(predictionCounter)
                if verbose_mode > 0:
                    print("start predict" + str(predictionCounter) )
                images_array = np.array(stored_frames)
                if verbose_mode > 0:
                    print(images_array.shape)
                images_array = np.reshape(images_array, (BATCH_SIZE, TARGET_FRAME, SIZE, CHANNEL))
                if verbose_mode > 0:
                    print("predicting model")
                model_results = model_tf_lstm.predict(images_array, verbose = 0)[0]
                if verbose_mode > 0:
                    print("model predicted")
                merged_prediction = list(zip(model_classes, model_results))
                
                if verbose_mode > 0:
                    print(merged_prediction)
                
                merged_prediction.sort(key=lambda a: a[1])
                
                if verbose_mode > 0:
                    print("Prediction :", merged_prediction[-1])
                predictionlist.append(merged_prediction[-1][0])
                indexedlist.append([idx+skip_image, merged_prediction[-1][0]])
                if verbose_mode > 0:
                    print(" ")
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

        errorpredict = []
        for letter in predictionlist:
            for tLetter in model_classes:
                if (letter == tLetter):
                    letter_mapping[model_classes.index(tLetter)] += 1

            if letter != TargetLetter:
                errorpredict.append(letter)

        print("Total Prediction: " + str( len(predictionlist) ))
        print("Prediction Mapping: " + str( letter_mapping ))
        print("Error Prediction: " + str( len(errorpredict) ))
        print("Error Rate: " + str( (len(errorpredict)/len(predictionlist))*100 ) )
        if view_prediction_list:
            print(str(predictionlist))
            print(str(indexedlist))

        final_result.append(str( len(predictionlist) ) + " || " +
                            str( letter_mapping ) + " || " +
                            str( len(errorpredict) ) + " || " +
                            str( (len(errorpredict)/len(predictionlist))*100 )
                            )
        
        total_predict = total_predict + len(predictionlist)
        total_error = total_error + len(errorpredict)
        final_letter_mapping.append(letter_mapping)

    for result in final_result:
        print(result)
        print(" ")
    print("Total Prediction: " + str( total_predict ))
    print("Error Prediction: " + str( total_error ))
    print("Error Rate: " + str( total_error/total_predict*100 ) )
    print("Correct Rate: "+ str( (1-(total_error/total_predict))*100 ))

    interpreter_path = interpreter_result_path + model_path + interpreter_config_path + result_suffix
    np.savetxt(interpreter_path+".csv", final_letter_mapping, delimiter=',', fmt='%d')
    print("Result saved: " + interpreter_path + ".csv")
    K.backend.clear_session()
    print ("Session Cleared")
    gc.collect()
    print ("Garbage Collector Enforced")
    del model_tf_lstm
    print ("Model Deleted")
    print ("Cycle for " + model_path + " Completed")