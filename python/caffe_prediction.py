#!/usr/bin/env python
# coding: utf8

# make sure that caffe is on the python path
CAFFE_ROOT = '/opt/caffe/caffe-master/'
import sys
sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe

import os
import glob
import cv2
import caffe
import numpy as np
from caffe.proto import caffe_pb2

# AVA
AVA_ROOT = '/Datasets/AVA/'
IMAGE_MEAN= AVA_ROOT + 'imagenet_mean.binaryproto'
DEPLOY = AVA_ROOT + 'initNetArch.deploy'
MODEL_FILE = AVA_ROOT + 'initModel_iter_16000.caffemodel'
# IMAGE_FILE = AVA_ROOT + "*jpg"

#caffe.set_mode_gpu()
caffe.set_mode_cpu()

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

'''
Image processing helper function
'''

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    return img

'''
Reading mean image, caffe model and its weights
'''
#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open(IMAGE_MEAN) as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

#Read model architecture and trained model's weights
net = caffe.Net(DEPLOY,
                MODEL_FILE,
                caffe.TEST)

#Define image transformers
print "Shape mean_array : ", mean_array.shape
print "Shape net : ", net.blobs['data'].data.shape
net.blobs['data'].reshape(1,        # batch size
                              3,         # channel
                              IMAGE_WIDTH, IMAGE_HEIGHT)  # image size
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

'''
Making predicitions
'''
#Reading image paths
test_img_paths = [img_path for img_path in glob.glob(IMAGE_FILE)]

#Making predictions
test_ids = []
preds = []
for img_path in test_img_paths:
    print img_path
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    print out
    print '-------'
