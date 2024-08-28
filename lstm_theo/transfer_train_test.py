import transfer_data_helper as tdh

#Train Data
tdh.transfer_data(
    classes=['-', 'A', 'E', 'I', 'O', 'U'],
    sourcedir="D:\Thesis\data_n2\\train",
    req_sdir="pos-landmark",
    dest_dir=".\\train_landmark\\"
)

#Test Data
tdh.transfer_data(
    classes=['-', 'A', 'E', 'I', 'O', 'U'],
    sourcedir="D:\Thesis\data_n2\\test",
    req_sdir="pos-landmark",
    dest_dir=".\\test_landmark\\"
)