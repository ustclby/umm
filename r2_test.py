import glob
import os
import numpy as np
import h5py
import caffe
import cv2

# x = np.load('/home/bylu/resnet_map/res_mean.npy', mmap_mode='r')
# rgb_mean = x[0, [2,1,0], :]
# batch_size = 16
root = '/opt/caffe/'
deploy = '/home/bylu/resnet_map/snap_20_deploy.prototxt'
caffe_model= '/home/bylu/resnet_map/snap_iter_15000.caffemodel'

caffe.set_mode_gpu()
net = caffe.Net(deploy, caffe_model, caffe.TEST)   # load model
# net.blobs['data'].reshape(batch_size, 3,data.shape[2],data.shape[3])

WD = r'/home/bylu/Test_h5/'
##WD = r'/Users/bylu/Downloads/'
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
        for i in range(len(data)):
            cnt = cnt + 1
            net.blobs['data'].data[...] = data[i]
            out = net.forward()
            predicts = out['prob'][0]
            p = np.argsort(-np.array(predicts))
            p1 = p[0]
            p2 = p[1]
            if p1 - labels[i] > 0.01 and p2 - labels[i] > 0.01:
                cntw = cntw + 1
                print(fn)
                print(i)
                img = np.transpose(data[i], (1, 2, 0))
                img_name = '/home/bylu/img/' + info[i][0:-4] + '_miscl_from_' + str(int(labels[i])) + '_to_' + str(pre) + '.jpg'
                print(img_name)
                cv2.imwrite(img_name, img)

accu = (cnt-cntw)/cnt
print(accu)
print('end')


    # for i in range(len(data)):
         #   data[i] = data[i]







#b = np.mean(b)
#g = np.mean(g)
#r = np.mean(r)
#z = x[0,:]