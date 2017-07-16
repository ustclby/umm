import glob
import os
import numpy as np
import h5py
import caffe
import cv2
import pylab as pl

r_mtx = np.load('/home/umm/r_mtx.npy', mmap_mode='r')
l_mtx = np.load('/home/umm/l_mtx.npy', mmap_mode='r')

pl.matshow(r_mtx)
pl.title('Confusion matrix of the roadClass classifier in Riverwalk')
pl.colorbar()
pl.savefig('/home/umm/roadclass_riverwalk.png')

pl.matshow(l_mtx)
pl.title('Confusion matrix of the laneCount classifier in Riverwalk')
pl.colorbar()
pl.savefig('/home/umm/laneCount_riverwalk.png')


precision_r = np.zeros((len(r_mtx[1,:])))
recall_r = np.zeros((len(r_mtx[:,1])))
precision_l = np.zeros((len(l_mtx[1,:])))
recall_l = np.zeros((len(l_mtx[:,1])))

# compute recall for r_mtx
for i in xrange(len(r_mtx[:,1])):
	recall_r[i] = float(r_mtx[i,i])/float(sum(r_mtx[i,:])) 
	print 'recall for roadclass ' + str(i+1) + 'is: ' + str(recall_r[i])

# compute recall for l_mtx
for i in xrange(len(l_mtx[:,1])):
	recall_l[i] = float(l_mtx[i,i])/float(sum(l_mtx[i,:])) 
	print 'recall for laneCount' + str(i+1) + 'is: ' + str(recall_l[i])

# compute precision for r_mtx
for i in xrange(len(r_mtx[1,:])):
	precision_r[i] = float(r_mtx[i,i])/float(sum(r_mtx[:,i])) 
	print 'precision for roadclass ' + str(i+1) + 'is: ' + str(precision_r[i])

# compute precision for l_mtx
for i in xrange(len(l_mtx[1,:])):
	precision_l[i] = float(l_mtx[i,i])/float(sum(l_mtx[:,i])) 
	print 'precision for laneCount ' + str(i+1) + 'is: ' + str(precision_l[i])

