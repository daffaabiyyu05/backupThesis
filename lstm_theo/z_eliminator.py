import transfer_data_helper as tdh
import numpy as np

fpath = "D:\Thesis\lstm_theo\\train_landmark\-\\000001.csv"
data = np.genfromtxt(fpath, delimiter=',')
#print(data)
data = np.delete(data, 2, 1)
#print(data)
print(data.shape)