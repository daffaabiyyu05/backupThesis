import os
import numpy as np
import cv2 as cv
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# grab the files
def file_grabber(root_dir : str, label : list, dataset : list):
    for fname in os.listdir(root_dir):
        full_path = os.path.join(root_dir, fname)
        dataset.append( (full_path, label) )

# grab the files and return a new list
def file_grabber_lister(root_dir : str) -> list:
    dataset = []
    for fname in os.listdir(root_dir):
        full_path = os.path.join(root_dir, fname)
        dataset.append(full_path)
    return dataset


# load images
def image_grayscale_loader(dataset : list) -> list:
    loaded_dataset = []
    print("|> loading text files")
    for fpath in tqdm(dataset):
        img = np.genfromtxt(fpath, delimiter=',')
        #img = cv.imread(fpath) #change to read csv file
        #img = cv.resize(img, (640, 480))
        #img = img[:, :, 0]
        #img = np.reshape(img, (640, 480, 1))
        #if np.sum(img) == 0 : # skip blank
        #    continue
        #img = ( img - np.min(img) ) / np.ptp(img)
        #img = np.float32(img)
        loaded_dataset.append( img )
    return loaded_dataset


# sliding window slicer 
def window_slicer(sequences : list, window_size = 1, offset_size = 1) -> list:
    sliced_dataset = []

    # if the window size is bigger or equal to sqeunce then return the sequnce
    if len(sequences) <= window_size:
        return sequences
    
    # if the seq bigger then cut it to smaller window
    for i in range(0, len(sequences) - window_size + 1, offset_size):
        sliced = sequences[ i : i + window_size ]
        sliced = np.array(sliced)
        sliced_dataset.append( sliced )
    return sliced_dataset

# all in one loader
def automatic_dataset_slicer_loader(
        source_dir : str, 
        label : list, 
        output_list_x : list,
        output_list_y : list, 
        window_size = 1, 
        offset_size = 1
    ):
    label = np.array(label)
    dataset = file_grabber_lister(source_dir)
    dataset = image_grayscale_loader(dataset)
    dataset = window_slicer(dataset, window_size, offset_size)
    print("|> adding to dataset")
    for img_seq in tqdm(dataset):
        output_list_x.append(img_seq)
        output_list_y.append(label)

def dataset_runtime_augmentation(input_tensor : tf.Tensor):
    
    input_tensor = tf.cast(input_tensor, dtype=tf.float32)
    
    # cursed div 255
    x = tf.math.divide(
        input_tensor,
        255
    )
    
    return x

def automatic_dataset_slicer_loader_auto_save(
        source_dir : str, 
        target_dir : str,
        label : list, 
        window_size = 1, 
        offset_size = 1
    ):
    label = np.array(label)
    os.makedirs(target_dir, exist_ok = True)
    
    dataset = file_grabber_lister(source_dir)
    #dataset = dataset[-128:]
    dataset = image_grayscale_loader(dataset)
    dataset = window_slicer(dataset, window_size, offset_size)
    
    print("|> saving to dataset")
    for counter in tqdm(range(len(dataset))):
        fpath = os.path.join(target_dir, f"{counter}.npz")
        np.savez_compressed(fpath, X = dataset[counter], Y = label )


class SequenceImagesDataset(tf.keras.utils.Sequence):

    def __init__(self, dataset_lst, batch_size : int, shuffle = True):
        # Initialization
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.dataset_files = dataset_lst

        self.datalen = len(self.dataset_files)
        self.indexes = np.arange(self.datalen)
        
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # get batch indexes from shuffled indexes
        batch_indexes = self.indexes[ (index * self.batch_size) : ( (index + 1) * self.batch_size) ]
        
        x_batch = []
        y_batch = []
        #print(batch_indexes)
        for index in batch_indexes:
            fname = self.dataset_files[index]
            nploaded = np.load(fname)
            
            npx = np.array(nploaded['X'])
            npy = np.array(nploaded['Y'])

            npx = np.float32(npx)
            npx = ( npx - np.min(npx) ) / np.ptp(npx)
            npx = np.float32(npx)

            x_batch.append(npx)
            y_batch.append(npy)
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        return x_batch, y_batch
    
    def __len__(self):
        # Denotes the number of batches per epoch
        return self.datalen // self.batch_size

    def on_epoch_end(self):
        
        # Updates indexes after each epoch
        self.indexes = np.arange(self.datalen)
        
        if self.shuffle:
            np.random.shuffle(self.indexes)


if __name__ == "__main__":
    print("hello")
    # split the files into training and testing
    dataset_files = []
    for root, d_names, f_names in os.walk('./Preprocessed'):
        for f in f_names:
            dataset_files.append( os.path.join(root, f) )
    
    from random import SystemRandom
    cryptorand = SystemRandom()
    cryptorand.shuffle(dataset_files)

    dataset_size = len(dataset_files)
    test_size = int(0.3 * dataset_size)


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

    print(a == b)
