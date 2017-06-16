import caffe # suppose caffe is already in the path of Python
import numpy as np
import sys

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( '/home/bylu/resnet_map/ResNet_mean.binaryproto' , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
arr.shape #  check the shape of arr
np.save('res_mean.npy', arr)
