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
#deploy = '/home/umm/resnet_map/snap_20_deploy.prototxt'
#caffe_model= '/home/umm/resnet_map/snap_iter_15000.caffemodel'
deploy = '/home/umm/resnet_map/multitask_resnet_tfs_deploy.prototxt'
caffe_model= '/home/umm/resnet_map/snap_iter_200000.caffemodel'

caffe.set_mode_gpu()
net = caffe.Net(deploy, caffe_model, caffe.TEST)   # load model
# net.blobs['data'].reshape(batch_size, 3,data.shape[2],data.shape[3])

WD = r'/home/umm/Test_h5/'

if not os.path.exists('/home/umm/img_laneCount'):
    os.makedirs('/home/umm/img_laneCount')
if not os.path.exists('/home/umm/img_roadClass/'):
    os.makedirs('/home/umm/img_roadClass/')
##WD = r'/Users/umm/Downloads/'
file_names = glob.glob(os.path.join(WD, '*.h5'))
cnt = 0
cntw_r = 0
cntw_l = 0
cnt0 = 0
for fn in file_names:
    with h5py.File(fn, 'r') as h5_file:
        data = h5_file['data'][0:]
        labels_roadClass = h5_file['label'][:, 13]
        labels_roadClass = list(labels_roadClass)
        info = h5_file['info'][0:]

        labels_laneCount = h5_file['label'][:, 8]
        labels_laneCount = list(labels_laneCount)




        # net.blobs['data'].reshape(len(data),3,data.shape[2],data.shape[3])
        cnt = cnt + len(data)
        for i in range(len(data)):
            net.blobs['data'].data[...] = data[i]
            out = net.forward()
            predicts_laneCount = out['laneCount'][0]
            predicts_roadClass = out['roadClass'][0]

            pre_l = predicts_laneCount.argmax()
            pre_r = predicts_roadClass.argmax()

            if abs(pre_r - labels_roadClass[i]) > 0.01:
                cntw_r += 1
                # print(fn)
                # print(i)
                img_r = np.transpose(data[i], (1, 2, 0))
                img_name_r = '/home/umm/img_roadClass/' + info[i][0:-4] + '_miscl_from_' + str(int(labels_roadClass[i])) + '_to_' + str(pre_r) + '.jpg'
                print(img_name_r)
                cv2.imwrite(img_name_r, img_r)
            if labels_laneCount[i] == 0:
                cnt0 += 1
                continue
            else:
                if abs(pre_l - labels_laneCount[i]) > 0.01:
                    cntw_l += 1
                # print(fn)
                # print(i)
                    img_l = np.transpose(data[i], (1, 2, 0))
                    img_name_l = '/home/umm/img_laneCount/' + info[i][0:-4] + '_miscl_from_' + str(int(labels_laneCount[i])) + '_to_' + str(pre_l) + '.jpg'
                    print(img_name_l)
                    cv2.imwrite(img_name_l, img_l)



print('accuracy_roadClass = ' + str(float(cnt-cntw_r)/float(cnt)))
print('accuracy_laneCount = ' + str(float(cnt-cntw_l-cnt0)/float(cnt-cnt0)))
print('end')


    # for i in range(len(data)):
         #   data[i] = data[i]







#b = np.mean(b)
#g = np.mean(g)
#r = np.mean(r)
#z = x[0,:]