import numpy as np
import os
from sklearn.model_selection import train_test_split
from random import SystemRandom

import helper_stuff as rhlp

PERCENTAGE = 0.3
model_window_size   = 30
dataset_offset_size = 3

# [A] dataset !
print("[A]")
rhlp.automatic_dataset_slicer_loader_auto_save(
    source_dir    = './train_landmark_norm/A',
    target_dir    = './Preprocessed_norm/A',
    label         = [1, 0, 0, 0, 0, 0],
    window_size   = model_window_size,
    offset_size   = dataset_offset_size
)

# [E] dataset !
print("[E]")
rhlp.automatic_dataset_slicer_loader_auto_save(
    source_dir    = './train_landmark_norm/E',
    target_dir    = './Preprocessed_norm/E', 
    label         = [0, 1, 0, 0, 0, 0],
    window_size   = model_window_size,
    offset_size   = dataset_offset_size
)

# [I] dataset !
print("[I]")
rhlp.automatic_dataset_slicer_loader_auto_save(
    source_dir    = './train_landmark_norm/I', 
    target_dir    = './Preprocessed_norm/I', 
    label         = [0, 0, 1, 0, 0, 0],
    window_size   = model_window_size,
    offset_size   = dataset_offset_size
)

# [O] dataset !
print("[O]")
rhlp.automatic_dataset_slicer_loader_auto_save(
    source_dir    = './train_landmark_norm/O',
    target_dir    = './Preprocessed_norm/O',  
    label         = [0, 0, 0, 1, 0, 0],
    window_size   = model_window_size,
    offset_size   = dataset_offset_size
)

# [U] dataset !
print("[U]")
rhlp.automatic_dataset_slicer_loader_auto_save(
    source_dir    = './train_landmark_norm/U',
    target_dir    = './Preprocessed_norm/U',  
    label         = [0, 0, 0, 0, 1, 0],
    window_size   = model_window_size,
    offset_size   = dataset_offset_size
)

# [-] dataset !
print("[-]")
rhlp.automatic_dataset_slicer_loader_auto_save(
    source_dir    = './train_landmark_norm/-',
    target_dir    = './Preprocessed_norm/-',   
    label         = [0, 0, 0, 0, 0, 1],
    window_size   = model_window_size,
    offset_size   = dataset_offset_size
)

# split the files into training and testing
dataset_files = []
for root, d_names, f_names in os.walk('./Preprocessed_norm'):
    for f in f_names:
        dataset_files.append( os.path.join(root, f) )

cryptorand = SystemRandom()
cryptorand.shuffle(dataset_files)

dataset_size = len(dataset_files)
test_size = int(PERCENTAGE * dataset_size)

training_set = dataset_files[test_size:]
testing_set  = dataset_files[:test_size]

print("TRAIN :", len(training_set))
print("TEST  :", len(testing_set))

import pickle
a = {'TRAIN': training_set, 'TEST' : testing_set}

with open('dataset_split.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('dataset_split.pickle', 'rb') as handle:
    b = pickle.load(handle)

print('sane :', a == b)

print("Complete !")
