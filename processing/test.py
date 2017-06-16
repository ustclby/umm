import glob
import os
import numpy as np
import h5py
import caffe
import cv2

# x = np.load('/home/umm/resnet_map/res_mean.npy', mmap_mode='r')
# rgb_mean = x[0, [2,1,0], :]
# batch_size = 16
root = '/opt/caffe/'
deploy = '/home/umm/resnet_map/snap_20_deploy.prototxt'
caffe_model= '/home/umm/resnet_map/snap_iter_15000.caffemodel'

caffe.set_mode_gpu()
net = caffe.Net(deploy, caffe_model, caffe.TEST)   # load model
# net.blobs['data'].reshape(batch_size, 3,data.shape[2],data.shape[3])

WD = r'/home/umm/Test_h5/'
##WD = r'/Users/umm/Downloads/'
file_names = glob.glob(os.path.join(WD, '*.h5'))
cnt = 0
cntw = 0
for fn in file_names:
    with h5py.File(fn, 'r') as h5_file:
        data = h5_file['data'][0:]
        labels = h5_file['label'][:, 1]
        labels = list(labels)
        info = h5_file['info'][0:]
        # net.blobs['data'].reshape(len(data),3,data.shape[2],data.shape[3])
        cnt = cnt + len(data)
        for i in range(len(data)):
            net.blobs['data'].data[...] = data[i]
            out = net.forward()
            predicts = out['prob'][0]
            pre = predicts.argmax()
            if abs(pre - labels[i]) > 0.01:
                cntw = cntw + 1
                # print(fn)
                # print(i)
                img = np.transpose(data[i], (1, 2, 0))
                img_name = '/home/umm/img/' + info[i][0:-4] + '_miscl_from_' + str(int(labels[i])) + '_to_' + str(pre) + '.jpg'
                print(img_name)
                cv2.imwrite(img_name, img)
print('accuracy = %5.3f',float(cnt-cntw)/float(cnt))
print('end')


    # for i in range(len(data)):
         #   data[i] = data[i]







#b = np.mean(b)
#g = np.mean(g)
#r = np.mean(r)
#z = x[0,:]