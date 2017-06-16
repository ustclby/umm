import glob
import os
import numpy as np
import h5py

x = np.load('/home/umm/resnet_map/res_mean.npy', mmap_mode='r')
rgb_mean = x[0, [2,1,0], :]


WD = r'/home/umm/Train_h5/'
file_names = glob.glob(os.path.join(WD, '*.h5'))
for fn in file_names:
    with h5py.File(fn, 'r') as h5_file:
        data = h5_file['data'][0:]
        labels = h5_file['label'][0:]
        data_shape = data.shape

        for i in range(len(data)):
            data[i] = (data[i] - rgb_mean) #* 0.00392157
            # data[i] = data[i] * 255
        data = data[:,[2,1,0],:]
    with h5py.File(fn, 'w') as f:
        f.create_dataset('data', data=data, compression="gzip")
        f.create_dataset('label', data=labels, compression="gzip")


WD = r'/home/umm/Test_h5/'
file_names = glob.glob(os.path.join(WD, '*.h5'))
for fn in file_names:
    with h5py.File(fn, 'r') as h5_file:
        data = h5_file['data'][0:]
        labels = h5_file['label'][0:]
        data_shape = data.shape

        for i in range(len(data)):
            data[i] = (data[i] - rgb_mean)# * 0.00392157
            # data[i] = data[i] * 255
        data = data[:,[2,1,0],:]
    with h5py.File(fn, 'w') as f:
        f.create_dataset('data', data=data, compression="gzip")
        f.create_dataset('label', data=labels, compression="gzip")
