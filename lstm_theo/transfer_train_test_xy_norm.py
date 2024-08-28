import transfer_data_helper as tdh

#Train Data
print("Train-Validation Set")
tdh.transfer_data_xy_norm(
    classes=['-', 'A', 'E', 'I', 'O', 'U'],
    sourcedir="D:\Thesis\lstm_theo\\train_landmark_xyz",
    req_sdir="",
    dest_dir=".\\train_landmark_norm\\"
)

#Test Data
print("Test Set")
tdh.transfer_data_xy_norm(
    classes=['-', 'A', 'E', 'I', 'O', 'U'],
    sourcedir="D:\Thesis\lstm_theo\\test_landmark_xyz",
    req_sdir="",
    dest_dir=".\\test_landmark_norm\\"
)

# import transfer_data_helper as tdh
# import numpy as np

# fpath = "D:\Thesis\lstm_theo\\train_landmark_xyz\-\\000109.csv"
# data = np.genfromtxt(fpath, delimiter=',')
# print(data)
# data = np.delete(data, 2, 1)
# print(data)
# print(data.shape)

# #START NORM
# x_arr = data[:,0]
# y_arr = data[:,1]

# x_norm = (x_arr - x_arr.min())/(x_arr.max() - x_arr.min())
# y_norm = (y_arr - y_arr.min())/(y_arr.max() - y_arr.min())

# data_norm = np.zeros([468,2])
# data_norm [:,0] = x_norm
# data_norm [:,1] = y_norm
# print(np.argwhere(np.isnan(data_norm)))
# print(np.argwhere(data_norm == 1))
# #END NORM
# print(data_norm.shape)