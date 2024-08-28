import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from random import SystemRandom
import shutil
from tqdm import tqdm

def iterate(rootdir, req_subdir, verbose = False):
    file_list = []
    for filename in glob.iglob(rootdir, recursive=True):
        if os.path.isfile(filename) and (req_subdir in filename): # filter dirs
            if verbose:
                print(filename)
            file_list.append(filename)
    return file_list

def transfer_data(classes, sourcedir, req_sdir, dest_dir, ext = '.csv', zerofill = 6, verbose = False):
    for current_class in classes:
        f_list = []
        print("Class: " + current_class)
        f_list = iterate(sourcedir + "\\" + current_class + "\\**", req_sdir)
        if verbose:
            print(f_list)
        data_index = 0
        l_dest_dir = dest_dir + current_class + "\\"
        print("|> copying data for Class " + current_class)
        for file in tqdm(f_list):
            data_index = data_index + 1
            shutil.copy2(file, l_dest_dir + str(data_index).zfill(zerofill) + ext)

def csv_z_eliminator(file, dest_file, delim = ','):
    data = np.genfromtxt(file, delimiter=delim)
    data = np.delete(data, 2, 1)
    np.savetxt(dest_file, data, delimiter=delim)

def csv_xy_norm(file, dest_file, delim = ',', NaNCheck = False, DropNaN = False):
    data = np.genfromtxt(file, delimiter=delim)
    data = np.delete(data, 2, 1)

    #START NORM
    x_arr = data[:,0]
    y_arr = data[:,1]

    x_norm = (x_arr - x_arr.min())/(x_arr.max() - x_arr.min())
    y_norm = (y_arr - y_arr.min())/(y_arr.max() - y_arr.min())

    data_norm = np.zeros([468,2])
    data_norm [:,0] = x_norm
    data_norm [:,1] = y_norm
    #END NORM
    #Check for any potential NaN when activated --> possible indicator of nonexistent data
    if NaNCheck and np.isnan(np.sum(data_norm)):
        print(np.argwhere(np.isnan(data_norm)))
        print(dest_file)

    #Work around NaN files by using the zero valued files instead if not dropped, else drop the file entirely
    if not np.isnan(np.sum(data_norm)):
        np.savetxt(dest_file, data_norm, delimiter=delim)
    else:
        if not DropNaN:
            np.savetxt(dest_file, data, delimiter=delim)


def transfer_data_zless(classes, sourcedir, req_sdir, dest_dir, ext = '.csv', zerofill = 6, verbose = False):
    for current_class in classes:
        f_list = []
        print("Class: " + current_class)
        f_list = iterate(sourcedir + "\\" + current_class + "\\**", req_sdir)
        if verbose:
            print(f_list)
        data_index = 0
        l_dest_dir = dest_dir + current_class + "\\"
        print("|> copying z-less data for Class " + current_class)
        for file in tqdm(f_list):
            data_index = data_index + 1
            csv_z_eliminator(file, l_dest_dir + str(data_index).zfill(zerofill) + ext)

def transfer_data_xy_norm(classes, sourcedir, req_sdir, dest_dir, ext = '.csv', zerofill = 6, verbose = False, checkForNaN = False):
    for current_class in classes:
        f_list = []
        print("Class: " + current_class)
        f_list = iterate(sourcedir + "\\" + current_class + "\\**", req_sdir, verbose)
        if verbose:
            print(f_list)
        data_index = 0
        l_dest_dir = dest_dir + current_class + "\\"
        print("|> copying normalized xy data for Class " + current_class)
        for file in tqdm(f_list):
            data_index = data_index + 1
            csv_xy_norm(file, l_dest_dir + str(data_index).zfill(zerofill) + ext, NaNCheck = checkForNaN)