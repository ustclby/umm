import glob
import os
import numpy as np
import h5py

WD = r'/home/bylu/Train_h5/'
#WD = r'/Users/bylu/Downloads/'
file_names = glob.glob(os.path.join(WD, '*.h5'))
new_label = []

for fn in file_names:
    with h5py.File(fn, 'r') as h5_file:
        labels = h5_file['label'][:, 1]
        labels = list(labels)
        new_label = new_label + labels
train_hist = np.histogram(new_label, bins=[0, 1, 2, 3, 4, 5, 6, 7])
train_hist = list(train_hist)
train_hist1 = list(train_hist[0])
train_hist2 = list(train_hist[1])
with open('hist.txt', 'w') as in_files:
    in_files.write(str(train_hist1))
    in_files.write(str(train_hist2))

####################################
WD2 = r'/home/bylu/test_h5/'
file_names2 = glob.glob(os.path.join(WD2, '*.h5'))
new_label2 = []

for fn in file_names2:
    with h5py.File(fn, 'r') as h5_file2:
        labels2 = h5_file2['label'][:,1]
        labels2 = list(labels2)
        new_label2 = new_label2 + labels2
test_hist = np.histogram(new_label2, bins=[0, 1, 2, 3, 4, 5, 6, 7])
test_hist = list(test_hist)
test_hist1 = list(test_hist[0])
test_hist2 = list(test_hist[1]
with open('hist.txt', 'a') as in_files2:
    in_files2.write('\n')
    in_files.write(str(test_hist1))
    in_files.write(str(test_hist2))
